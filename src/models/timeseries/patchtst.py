import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

from src.models.base import BaseModel

# Half-width of the uniform initialization of the learned positional encoding (reference code).
POSITION_INIT_RANGE = 0.02


class ResidualAttention(nn.Module):
    """Multi-head self-attention that adds the previous layer's pre-softmax scores
    (RealFormer, https://arxiv.org/abs/2012.11747), as in the PatchTST reference code."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, prev_scores: Tensor | None) -> tuple[Tensor, Tensor]:
        """(b, l, d), (b, h, l, l) | None -> (b, l, d), scores (b, h, l, l)"""
        q, k, v = (
            rearrange(proj(x), "b l (h e) -> b h l e", h=self.n_heads)
            for proj in (self.query, self.key, self.value)
        )
        scores = einsum(q, k, "b h l e, b h m e -> b h l m") * self.scale
        if prev_scores is not None:
            scores = scores + prev_scores
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = rearrange(einsum(weights, v, "b h l m, b h m e -> b h l e"), "b h l e -> b l (h e)")
        return self.out_dropout(self.out(out)), scores


class EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer with residual attention and a GELU feed-forward block.

    LayerNorm replaces the reference code's BatchNorm, so the layer computes the same function
    in train and eval mode and for any batch size.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.attention = ResidualAttention(d_model, n_heads, attn_dropout, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # residual dropout of both sublayers

    def forward(self, x: Tensor, prev_scores: Tensor | None) -> tuple[Tensor, Tensor]:
        """(b, l, d), (b, h, l, l) | None -> (b, l, d), scores (b, h, l, l)"""
        out, scores = self.attention(x, prev_scores)
        x = self.attention_norm(x + self.dropout(out))
        x = self.ff_norm(x + self.dropout(self.ff(x)))
        return x, scores


class PatchTST(BaseModel):
    """PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    (Nie et al., ICLR 2023), https://arxiv.org/abs/2211.14730.
    Reference code: https://github.com/yuqinie98/PatchTST

    Every (node, channel) series is forecast on its own (channel independence). The series is
    padded at its end by replicating its last value `stride` times and split into overlapping
    patches, which a Transformer encoder with residual attention processes as tokens; a linear
    head maps the flattened tokens to the horizon. With `individual` every node has its own head.

    JEPA split: the encoder is patching, patch embedding and the Transformer encoder; its latent
    is (b, n, c, l, d) with l patches of width d per series. The predictor mixes the patches of
    each series. The decoder is the flatten head.

    Deviations from the reference code: LayerNorm instead of BatchNorm in the encoder, and no
    RevIN inside the model (the trainer's `train.instance_norm` normalizes every input window).

    custom_time_features: channel independence leaves no way to use time features (a token is
    a patch of one series), so the model ignores them instead of receiving the shared encoding.
    """

    custom_time_features = True

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        attn_dropout: float,
        individual: bool,
        **common,
    ):
        super().__init__(**common)
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if self.out_channels != self.in_channels:
            raise ValueError("PatchTST forecasts every input channel: out_channels == in_channels")
        if patch_len > self.his_len + stride:
            raise ValueError(f"patch_len ({patch_len}) exceeds the padded his_len")
        self.patch_len = patch_len
        self.stride = stride
        # Patches of the series padded with `stride` steps at its end.
        num_patches = (self.his_len + stride - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(
            torch.empty(num_patches, d_model).uniform_(-POSITION_INIT_RANGE, POSITION_INIT_RANGE)
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout) for _ in range(e_layers)
        )
        self.predictor = self.make_predictor(num_tokens=num_patches, dim=d_model)

        # Head: linear map of the flattened tokens (d_model * num_patches) to the horizon, one
        # per node with `individual`; initialized like nn.Linear.
        head_in = d_model * num_patches
        nodes = (self.num_nodes,) if individual else ()
        self.head_weight_pattern = (
            "b n c e, n p e -> b p n c" if individual else "b n c e, p e -> b p n c"
        )
        self.head_bias_pattern = "n p -> p n 1" if individual else "p -> p 1 1"
        bound = 1 / math.sqrt(head_in)
        self.head_weight = nn.Parameter(
            torch.empty(*nodes, self.pred_len, head_in).uniform_(-bound, bound)
        )
        self.head_bias = nn.Parameter(torch.empty(*nodes, self.pred_len).uniform_(-bound, bound))

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """x: (b, t, n, c) -> latent (b, n, c, l, d), l = number of patches."""
        b, _, n, _ = x.shape
        series = rearrange(x, "b t n c -> (b n c) t")  # every series is encoded on its own
        series = F.pad(series, (0, self.stride), mode="replicate")
        # Sliding windows over time: (b n c, l, patch_len).
        patches = series.unfold(-1, self.patch_len, self.stride)
        tokens = self.dropout(self.patch_embedding(patches) + self.position)  # (b n c, l, d)

        scores = None
        for layer in self.layers:
            tokens, scores = layer(tokens, scores)
        return rearrange(tokens, "(b n c) l d -> b n c l d", b=b, n=n)

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, c, l, d) -> (b, n, c, l, d), mixing the patches of each series."""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """z: (b, n, c, l, d) -> (b, pred_len, n, c)"""
        # Flatten in the reference order: width-major, then patches.
        flat = rearrange(z, "b n c l d -> b n c (d l)")
        out = einsum(flat, self.head_weight, self.head_weight_pattern)
        return out + rearrange(self.head_bias, self.head_bias_pattern)
