import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from src.models.base import BaseModel


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention along the second-to-last axis of (b, s, l, d)."""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """(b, s, l, d) -> (b, s, l, d); every one of the s sequences attends over its l steps."""
        q, k, v = (
            rearrange(proj(x), "b s l (h e) -> b s h l e", h=self.num_heads)
            for proj in (self.query, self.key, self.value)
        )
        out = F.scaled_dot_product_attention(q, k, v)  # (b, s, h, l, e), scale 1/sqrt(e)
        return self.out(rearrange(out, "b s h l e -> b s l (h e)"))


class SelfAttentionLayer(nn.Module):
    """Post-norm Transformer encoder layer: attention and a ReLU feed-forward block, each with
    dropout, a residual connection and LayerNorm."""

    def __init__(self, model_dim: int, feed_forward_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(b, s, l, d) -> (b, s, l, d), attention along l."""
        x = self.norm1(x + self.dropout1(self.attention(x)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))


class STAEformer(BaseModel):
    """STAEformer (Liu et al., CIKM 2023), "STAEformer: Spatio-Temporal Adaptive Embedding Makes
    Vanilla Transformer SOTA for Traffic Forecasting", https://arxiv.org/abs/2308.10425.

    Reference code: https://github.com/XDZhelheim/STAEformer

    Every (step, node) token concatenates an input projection, time-of-day and day-of-week
    embeddings, a node embedding and a learned spatio-temporal adaptive embedding of shape
    (his_len, n). Plain Transformer layers attend first along time (per node), then along nodes
    (per step); a linear head maps the result to the horizon. Each embedding is disabled by
    setting its width to 0; a time embedding is also dropped when the dataset lacks its feature.

    JEPA split: the encoder is the embedding and both Transformer stacks; its latent is the
    token sequence of every node, (b, n, his_len, model_dim), so the predictor mixes the steps of
    each node. The decoder is the output projection.

    custom_time_features: the temporal embedding is one of the paper's core contributions. As
    in the official code, the time-of-day and day-of-week features of the window's steps index
    their own learned tables (day of month / year are not used), whose widths are part of the
    token layout, so the shared time encoding does not fit.
    """

    custom_time_features = True

    def __init__(
        self,
        input_embedding_dim: int,
        tod_embedding_dim: int,
        dow_embedding_dim: int,
        spatial_embedding_dim: int,
        adaptive_embedding_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        num_layers: int,
        use_mixed_proj: bool,
        dropout: float,
        **common,
    ):
        super().__init__(**common)
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(self.in_channels, input_embedding_dim)
        # One table per embedded time feature, sized by its number of distinct values
        # (time_of_day: steps per day, day_of_week: 7); x_mark column of each feature.
        widths = {"time_of_day": tod_embedding_dim, "day_of_week": dow_embedding_dim}
        columns = list(self.time_feature_sizes)
        self.time_embeddings = nn.ModuleDict()
        self.time_columns: dict[str, int] = {}
        for name, width in widths.items():
            if width > 0 and name in self.time_feature_sizes:
                self.time_embeddings[name] = nn.Embedding(self.time_feature_sizes[name], width)
                self.time_columns[name] = columns.index(name)
        time_dim = sum(table.embedding_dim for table in self.time_embeddings.values())
        self.model_dim = (
            input_embedding_dim + time_dim + spatial_embedding_dim + adaptive_embedding_dim
        )

        self.node_embedding = None
        if spatial_embedding_dim > 0:
            self.node_embedding = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(self.num_nodes, spatial_embedding_dim))
            )
        # Tied to the window length: input and target windows both have his_len steps and
        # share it, each step of a window getting the same embedding whichever window it is.
        self.adaptive_embedding = None
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.his_len, self.num_nodes, adaptive_embedding_dim)
                )
            )

        def layers() -> nn.ModuleList:
            return nn.ModuleList(
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            )

        self.temporal_layers = layers()
        self.spatial_layers = layers()

        # Mixes the his_len steps of every node, each step a model_dim-wide token.
        self.predictor = self.make_predictor(num_tokens=self.his_len, dim=self.model_dim)

        if use_mixed_proj:
            # One linear map from the whole (time, feature) history of a node to its horizon.
            self.output_proj = nn.Linear(
                self.his_len * self.model_dim, self.pred_len * self.out_channels
            )
        else:
            # A linear map over time per feature, then over features per step.
            self.temporal_proj = nn.Linear(self.his_len, self.pred_len)
            self.output_proj = nn.Linear(self.model_dim, self.out_channels)

    def _embed(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """x: (b, t, n, c), x_mark: (b, t, f) -> tokens (b, t, n, model_dim)"""
        b, t, n, _ = x.shape
        features = [self.input_proj(x)]
        for name, table in self.time_embeddings.items():
            size = table.num_embeddings
            # features are index / size; rounding absorbs float error, clamping guards 1.0
            index = (x_mark[..., self.time_columns[name]] * size).round().long()  # (b, t)
            index = index.clamp(max=size - 1)
            features.append(repeat(table(index), "b t d -> b t n d", n=n))
        if self.node_embedding is not None:
            features.append(self.node_embedding.expand(b, t, -1, -1))
        if self.adaptive_embedding is not None:
            features.append(self.adaptive_embedding.expand(b, -1, -1, -1))
        return torch.cat(features, dim=-1)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """x: (b, t, n, c), x_mark: (b, t, f) -> latent (b, n, t, model_dim), t = his_len."""
        h = rearrange(self._embed(x, x_mark), "b t n d -> b n t d")
        for layer in self.temporal_layers:
            h = layer(h)  # attention along t, per node
        h = rearrange(h, "b n t d -> b t n d")
        for layer in self.spatial_layers:
            h = layer(h)  # attention along n, per step
        return rearrange(h, "b t n d -> b n t d")

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, t, d) -> (b, n, t, d), mixing the steps of each node."""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """z: (b, n, t, model_dim) -> (b, p, n, out_channels); only z is read."""
        if self.use_mixed_proj:
            out = self.output_proj(rearrange(z, "b n t d -> b n (t d)"))  # (b, n, p * c)
            return rearrange(out, "b n (p c) -> b p n c", p=self.pred_len)
        h = self.temporal_proj(rearrange(z, "b n t d -> b d n t"))  # (b, d, n, p)
        return self.output_proj(rearrange(h, "b d n p -> b p n d"))
