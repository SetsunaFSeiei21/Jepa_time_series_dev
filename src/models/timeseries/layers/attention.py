import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

# Feed-forward activations selectable in the yaml.
ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu}
# Inputs an inner attention mechanism may read (its `inputs`), each with its own projection.
INPUTS = ("queries", "keys", "values")


def fit_length(x: Tensor, length: int) -> Tensor:
    """Zero-pads or truncates the time axis to `length`: (b, s, ...) -> (b, length, ...)"""
    missing = length - x.shape[1]
    if missing <= 0:
        return x[:, :length]
    return torch.cat([x, x.new_zeros(x.shape[0], missing, *x.shape[2:])], dim=1)


class ProjectedAttention(nn.Module):
    """Multi-head input and output projections around an inner attention mechanism.

    The inner module lists in `inputs` which of queries / keys / values it reads; only those get
    a projection. It receives them split into heads, (b, t, h, e), and returns (b, t, h, e).
    The inner module may be shared between several layers (FEDformer), the projections never are.
    """

    def __init__(self, inner: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.inner = inner
        self.n_heads = n_heads
        for name in INPUTS:
            projection = nn.Linear(d_model, d_model) if name in inner.inputs else None
            setattr(self, f"{name}_projection", projection)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        """x: (b, t, d) queries, context: (b, s, d) keys and values (x if None) -> (b, t, d)"""
        context = x if context is None else context
        sources = {"queries": x, "keys": context, "values": context}
        heads = {
            name: rearrange(
                getattr(self, f"{name}_projection")(sources[name]),
                "b t (h e) -> b t h e",
                h=self.n_heads,
            )
            for name in self.inner.inputs
        }
        out = self.inner(**heads)
        return self.out_projection(rearrange(out, "b t h e -> b t (h e)"))


class FullAttention(nn.Module):
    """Unmasked scaled dot-product attention with dropout on the attention weights.

    The dropout is an `nn.Dropout` module (not the `dropout_p` of a fused kernel), so that it
    follows the module's mode and can be switched off on its own (JEPA's online encoder).
    """

    inputs: ClassVar[tuple[str, ...]] = ("queries", "keys", "values")

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """queries (b, t, h, e), keys / values (b, s, h, e) -> (b, t, h, e)"""
        scores = einsum(queries, keys, "b t h e, b s h e -> b h t s") / math.sqrt(queries.shape[-1])
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return einsum(weights, values, "b h t s, b s h e -> b t h e")
