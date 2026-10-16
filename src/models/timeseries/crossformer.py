import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn

from src.models.base import BaseModel


class SegmentAttention(nn.Module):
    """Multi-head scaled dot-product attention with dropout on the attention weights, as in
    the reference code (`AttentionLayer` around `FullAttention`).

    `num_keys` is the number of keys the layer always attends over. With a single key the
    softmax is 1 whatever the queries and keys are, so query and key projections could never
    act (they would only ever receive a zero gradient) and are not built; the layer then
    computes `out(dropout(1) * value(keys))`, exactly what the reference computes.
    """

    def __init__(self, d_model: int, n_heads: int, num_keys: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.query = nn.Linear(d_model, d_model) if num_keys > 1 else None
        self.key = nn.Linear(d_model, d_model) if num_keys > 1 else None
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """queries (b, t, d), keys = values (b, s, d) -> (b, t, d)"""
        v = rearrange(self.value(keys), "b s (h e) -> b h s e", h=self.n_heads)
        if self.query is None:
            # a single key: the scores do not matter, the softmax is 1
            scores = v.new_zeros(v.shape[0], self.n_heads, queries.shape[1], 1)
        else:
            q = rearrange(self.query(queries), "b t (h e) -> b h t e", h=self.n_heads)
            k = rearrange(self.key(keys), "b s (h e) -> b h s e", h=self.n_heads)
            scores = einsum(q, k, "b h t e, b h s e -> b h t s") * self.scale
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = einsum(weights, v, "b h t s, b h s e -> b h t e")
        return self.out(rearrange(out, "b h t e -> b t (h e)"))


class TwoStageAttention(nn.Module):
    """Two-Stage Attention (TSA) layer on segment tokens (b, n, l, d).

    Cross-time stage: self-attention over the l segments of every variate. Cross-dimension
    stage: `factor` learned routers per segment gather messages from all n variates and
    distribute them back, so the cost is linear in n. Post-norm residual blocks, as in the
    reference code.
    """

    def __init__(
        self,
        num_segments: int,
        num_variates: int,
        factor: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.time_attention = SegmentAttention(d_model, n_heads, num_segments, dropout)
        self.dim_sender = SegmentAttention(d_model, n_heads, num_variates, dropout)
        self.dim_receiver = SegmentAttention(d_model, n_heads, factor, dropout)
        self.router = nn.Parameter(torch.randn(num_segments, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff_norm = nn.LayerNorm(d_model)
        self.dim_norm = nn.LayerNorm(d_model)
        self.dim_ff_norm = nn.LayerNorm(d_model)
        self.time_ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dim_ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """(b, n, l, d) -> (b, n, l, d)"""
        b = x.shape[0]
        # cross-time stage, every variate on its own: (b n, l, d)
        h = rearrange(x, "b n l d -> (b n) l d")
        h = self.time_norm(h + self.dropout(self.time_attention(h, h)))
        h = self.time_ff_norm(h + self.dropout(self.time_ff(h)))

        # cross-dimension stage, every segment on its own: (b l, n, d), routers (b l, r, d)
        h = rearrange(h, "(b n) l d -> (b l) n d", b=b)
        routers = repeat(self.router, "l r d -> (b l) r d", b=b)
        buffer = self.dim_sender(routers, h)
        h = self.dim_norm(h + self.dropout(self.dim_receiver(h, buffer)))
        h = self.dim_ff_norm(h + self.dropout(self.dim_ff(h)))
        return rearrange(h, "(b l) n d -> b n l d", b=b)


class SegmentMerging(nn.Module):
    """Merges every `win_size` adjacent segments into one (LayerNorm + linear map of their
    concatenation): (b, n, l, d) -> (b, n, ceil(l / win_size), d).

    When l is not a multiple of `win_size`, the last `missing` segments are appended again, as
    in the reference code. A scale with fewer than `missing` segments (where the reference code
    fails) repeats its first segment for the indices before it.
    """

    def __init__(self, d_model: int, win_size: int):
        super().__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(win_size * d_model)
        self.linear = nn.Linear(win_size * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """(b, n, l, d) -> (b, n, ceil(l / win_size), d)"""
        length = x.shape[2]
        missing = -length % self.win_size
        repeated = torch.arange(length - missing, length, device=x.device).clamp(min=0)
        x = torch.cat([x, x[:, :, repeated]], dim=2)
        x = rearrange(x, "b n (l w) d -> b n l (w d)", w=self.win_size)
        return self.linear(self.norm(x))


class ScaleBlock(nn.Module):
    """One encoder scale: optional segment merging followed by `depth` TSA layers."""

    def __init__(self, win_size: int, depth: int, num_segments: int, **tsa):
        super().__init__()
        self.merge = SegmentMerging(tsa["d_model"], win_size) if win_size > 1 else nn.Identity()
        self.layers = nn.ModuleList(TwoStageAttention(num_segments, **tsa) for _ in range(depth))

    def forward(self, x: Tensor) -> Tensor:
        """(b, n, l, d) -> (b, n, l', d), l' = ceil(l / win_size)"""
        x = self.merge(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    """TSA over the decoder's segments, cross-attention to one encoder scale, an MLP, and a
    linear map of every decoder segment to `seg_len` forecast steps."""

    def __init__(
        self,
        seg_len: int,
        out_segments: int,
        scale_segments: int,
        dropout: float,
        **tsa,
    ):
        super().__init__()
        d_model = tsa["d_model"]
        self.self_attention = TwoStageAttention(out_segments, dropout=dropout, **tsa)
        self.cross_attention = SegmentAttention(d_model, tsa["n_heads"], scale_segments, dropout)
        self.dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.forecast = nn.Linear(d_model, seg_len)

    def forward(self, x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        """x (b, n, l_out, d), encoder scale (b, n, l_i, d)
        -> x (b, n, l_out, d), forecast of this layer (b, n, l_out, seg_len)"""
        b = x.shape[0]
        h = rearrange(self.self_attention(x), "b n l d -> (b n) l d")
        context = rearrange(scale, "b n l d -> (b n) l d")
        h = self.cross_norm(h + self.dropout(self.cross_attention(h, context)))
        h = self.mlp_norm(h + self.mlp(h))
        h = rearrange(h, "(b n) l d -> b n l d", b=b)
        return h, self.forecast(h)


class Crossformer(BaseModel):
    """Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time
    Series Forecasting (Zhang & Yan, ICLR 2023), https://openreview.net/forum?id=vSVLM2j9eie.
    Reference code: https://github.com/Thinklab-SJTU/Crossformer

    Every variate (a node's channel) is cut into segments of `seg_len` steps, each embedded by
    a linear map plus a learned position of its own (Dimension-Segment-Wise embedding). A
    hierarchical encoder of Two-Stage-Attention blocks, merging `win_size` segments between
    blocks, yields one representation per scale; decoder layer i starts from learned
    positions of the forecast segments, cross-attends to scale i and forecasts the horizon;
    the forecasts of all layers are summed. Inputs whose length is not a multiple of
    `seg_len` are padded at the front with their first value, as in the reference code.

    Latent: the list of scales, scale i of shape (b, n * c, l_i, d_model) with l_i segments;
    one latent predictor per scale mixes its segments.

    custom_time_features: Crossformer uses no time features. The decoder is driven by learned
    position embeddings alone, and the embedding works on segments of one variate, so time
    channels appended to every node would become extra variates rather than features of the
    series; the model therefore ignores x_mark and y_mark.
    """

    custom_time_features = True

    def __init__(
        self,
        seg_len: int,
        win_size: int,
        factor: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        block_depth: int,
        dropout: float,
        baseline: bool,
        **common,
    ):
        super().__init__(**common)
        assert self.in_channels == self.out_channels, "every input variate is forecast"
        self.seg_len = seg_len
        self.baseline = baseline
        num_variates = self.num_nodes * self.in_channels
        in_segments = math.ceil(self.his_len / seg_len)
        out_segments = math.ceil(self.pred_len / seg_len)
        self.front_padding = in_segments * seg_len - self.his_len

        # Dimension-Segment-Wise embedding, with a learned position per variate and segment
        self.embedding = nn.Linear(seg_len, d_model)
        self.position = nn.Parameter(torch.randn(num_variates, in_segments, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        tsa = dict(
            num_variates=num_variates, factor=factor, d_model=d_model, n_heads=n_heads, d_ff=d_ff
        )
        # segments of every scale: the embedding, then each block after merging
        scale_segments = [in_segments]
        self.encoder_blocks = nn.ModuleList()
        for i in range(e_layers):
            merge = win_size if i > 0 else 1
            scale_segments.append(math.ceil(scale_segments[-1] / merge))
            self.encoder_blocks.append(
                ScaleBlock(merge, block_depth, scale_segments[-1], dropout=dropout, **tsa)
            )

        self.predictors = nn.ModuleList(
            self.make_predictor(num_tokens=segments, dim=d_model) for segments in scale_segments
        )

        self.decoder_position = nn.Parameter(torch.randn(num_variates, out_segments, d_model))
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(seg_len, out_segments, segments, dropout, **tsa)
            for segments in scale_segments
        )

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> list[Tensor]:
        """(b, t, n, c) -> scales [(b, n c, l_i, d)]"""
        series = rearrange(x, "b t n c -> b (n c) t")
        series = F.pad(series, (self.front_padding, 0), mode="replicate")
        segments = rearrange(series, "b n (l s) -> b n l s", s=self.seg_len)
        h = self.pre_norm(self.embedding(segments) + self.position)  # (b, n c, l_0, d)
        scales = [h]
        for block in self.encoder_blocks:
            h = block(h)
            scales.append(h)
        return scales

    def predict(self, z: list[Tensor]) -> list[Tensor]:
        """scales [(b, n, l_i, d)] -> scales of the following window, same shapes"""
        return [predictor(scale) for predictor, scale in zip(self.predictors, z, strict=True)]

    def decode(
        self,
        z: list[Tensor],
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """scales [(b, n c, l_i, d)] -> (b, pred_len, n, c)"""
        h = repeat(self.decoder_position, "n l d -> b n l d", b=x.shape[0])
        forecast = 0
        for layer, scale in zip(self.decoder_layers, z, strict=True):
            h, layer_forecast = layer(h, scale)
            forecast = forecast + layer_forecast  # (b, n c, l_out, seg_len)
        y = rearrange(forecast, "b (n c) l s -> b (l s) n c", c=self.out_channels)
        y = y[:, : self.pred_len]
        if self.baseline:
            y = y + x.mean(dim=1, keepdim=True)
        return y

    def latent_tokens(self, z: list[Tensor]) -> Tensor:
        """scales [(b, n, l_i, d)] -> (b, sum_i n l_i, d)"""
        return torch.cat([rearrange(scale, "b n l d -> b (n l) d") for scale in z], dim=1)
