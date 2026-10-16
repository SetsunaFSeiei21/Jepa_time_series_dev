import torch
from einops import einsum, rearrange
from torch import Tensor, nn

from src.models.base import BaseModel
from src.models.layers import DiffusionConv

# Learned positional embeddings are initialized like every embedding table of the official STTN
# code (BERT's initializer): a normal distribution with this standard deviation, truncated at
# POSITION_EMBEDDING_TRUNCATION standard deviations. They then start much smaller than the
# projected input, which carries the signal.
POSITION_EMBEDDING_STD = 0.02
POSITION_EMBEDDING_TRUNCATION = 2.0


def position_embedding(length: int, dim: int) -> nn.Parameter:
    """Learned embedding (length, dim) added to the tokens of an attention axis."""
    embedding = nn.Parameter(torch.empty(length, dim))
    bound = POSITION_EMBEDDING_TRUNCATION * POSITION_EMBEDDING_STD
    nn.init.trunc_normal_(embedding, std=POSITION_EMBEDDING_STD, a=-bound, b=bound)
    return embedding


class Attention(nn.Module):
    """Multi-head self-attention over the second-to-last axis of (..., l, d), optionally causal.

    With `causal=True` the scores of future positions are set to -inf before the (single)
    softmax, so position l attends to positions <= l only and its weights sum to one over them.
    """

    def __init__(self, dim: int, heads: int, dropout: float, causal: bool):
        super().__init__()
        assert dim % heads == 0, f"hidden_channels {dim} must be divisible by heads {heads}"
        self.heads = heads
        self.causal = causal
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(..., l, d) -> (..., l, d)"""
        q, k, v = rearrange(
            self.qkv(x), "... l (three h e) -> three ... h l e", three=3, h=self.heads
        )
        scores = einsum(q, k, "... h l e, ... h m e -> ... h l m") * self.scale
        if self.causal:
            length = x.shape[-2]
            future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        h = einsum(scores.softmax(dim=-1), v, "... h l m, ... h m e -> ... h l e")
        h = rearrange(h, "... h l e -> ... l (h e)")
        return self.dropout(self.proj(h))


def feed_forward(dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Pre-norm position-wise MLP: (..., d) -> (..., d)."""
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class SpatialLayer(nn.Module):
    """Spatial self-attention and feed-forward, plus a graph diffusion of the layer input."""

    def __init__(
        self, dim: int, heads: int, mlp_dim: int, num_supports: int, steps: int, dropout: float
    ):
        super().__init__()
        self.attention = Attention(dim, heads, dropout, causal=False)
        self.feed_forward = feed_forward(dim, mlp_dim, dropout)
        self.graph_conv = DiffusionConv(dim, dim, num_supports, steps, dropout)

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        """x: (b, t, n, d), supports: (b, k, n, n) -> (b, t, n, d)"""
        h = self.attention(x) + x
        h = self.feed_forward(h) + h
        return self.graph_conv(x, supports) + h


class TemporalLayer(nn.Module):
    """Causal temporal self-attention and feed-forward."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.attention = Attention(dim, heads, dropout, causal=True)
        self.feed_forward = feed_forward(dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """(b, n, t, d) -> (b, n, t, d)"""
        h = self.attention(x) + x
        return self.feed_forward(h) + h


class STTNBlock(nn.Module):
    """Spatial transformer, temporal transformer and layer norm, with a residual connection.

    Spatial attention mixes the nodes of each time step, temporal attention the (past) time
    steps of each node; each transformer first adds its learned positional embedding.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_nodes: int,
        his_len: int,
        num_supports: int,
        diffusion_steps: int,
        dropout: float,
    ):
        super().__init__()
        self.spatial_embedding = position_embedding(num_nodes, dim)
        self.temporal_embedding = position_embedding(his_len, dim)
        self.spatial_layers = nn.ModuleList(
            SpatialLayer(dim, heads, mlp_dim, num_supports, diffusion_steps, dropout)
            for _ in range(depth)
        )
        self.temporal_layers = nn.ModuleList(
            TemporalLayer(dim, heads, mlp_dim, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        """x: (b, t, n, d), supports: (b, k, n, n) -> (b, t, n, d)"""
        h = x + self.spatial_embedding
        for layer in self.spatial_layers:
            h = layer(h, supports)
        h = rearrange(h, "b t n d -> b n t d") + self.temporal_embedding
        for layer in self.temporal_layers:
            h = layer(h)
        return rearrange(self.norm(h), "b n t d -> b t n d") + x


class STTN(BaseModel):
    """Spatial-Temporal Transformer Networks for Traffic Flow Forecasting (Xu et al., 2020),
    https://arxiv.org/abs/2001.02908.

    Official code (TensorFlow): https://github.com/xumingxingsjtu/STTN. The architecture here is
    the PyTorch STTN of LargeST (https://github.com/liuxu77/LargeST, src/models/sttn.py), with
    LayerNorm instead of BatchNorm, a single softmax in the causal temporal attention and the
    positional embedding initialization of the official code; see docs/models/sttn.md.

    encode:  start projection and ST blocks -> latent (b, n, t, hidden_channels)
    predict: LatentPredictor mixing the t steps of each node
    decode:  two-layer head on the last step of each node -> (b, pred_len, n, out_channels)
    """

    def __init__(
        self,
        hidden_channels: int,
        end_channels: int,
        blocks: int,
        layers_per_block: int,
        heads: int,
        mlp_expand: int,
        diffusion_steps: int,
        dropout: float,
        num_supports: int,
        **common,
    ):
        super().__init__(**common)
        self.start = nn.Linear(self.in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            STTNBlock(
                hidden_channels,
                layers_per_block,
                heads,
                hidden_channels * mlp_expand,
                self.num_nodes,
                self.his_len,
                num_supports,
                diffusion_steps,
                dropout,
            )
            for _ in range(blocks)
        )
        self.predictor = self.make_predictor(num_tokens=self.his_len, dim=hidden_channels)
        self.end = nn.Sequential(
            nn.Linear(hidden_channels, end_channels),
            nn.ReLU(),
            nn.Linear(end_channels, self.pred_len * self.out_channels),
        )

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        """x: (b, t, n, c), x_mark: (b, t, f), supports: (b, k, n, n) -> (b, n, t, d)"""
        h = self.start(self.add_time(x, x_mark))  # (b, t, n, d)
        for block in self.blocks:
            h = block(h, supports)
        return rearrange(h, "b t n d -> b n t d")

    def predict(self, z: Tensor) -> Tensor:
        """(b, n, t, d) -> (b, n, t, d), mixing the time steps of each node"""
        return self.predictor(z)

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        """z: (b, n, t, d) -> (b, pred_len, n, out_channels)"""
        # The last step is the only one whose causal temporal attention covered the whole window.
        h = self.end(z[:, :, -1])  # (b, n, p * c)
        return rearrange(h, "b n (p c) -> b p n c", c=self.out_channels)
