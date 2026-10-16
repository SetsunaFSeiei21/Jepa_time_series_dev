import torch
from einops import einsum, rearrange, repeat
from torch import Tensor, nn


class DiffusionConv(nn.Module):
    """Graph diffusion convolution (DCRNN, Graph WaveNet, STTN).

    Concatenates x and A^1 x .. A^steps x for every support A, then mixes the channels with a
    linear layer. Works on any layout with nodes and channels last.
    """

    def __init__(self, in_dim: int, out_dim: int, num_supports: int, steps: int, dropout: float):
        super().__init__()
        self.steps = steps
        self.mix = nn.Linear((steps * num_supports + 1) * in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        """x: (b, ..., n, d), supports: (b, k, n, n) -> (b, ..., n, d_out)"""
        h = repeat(x, "b ... n d -> b k ... n d", k=supports.shape[1])
        diffused = []
        for _ in range(self.steps):
            h = einsum(h, supports, "b k ... n d, b k n m -> b k ... m d")
            diffused.append(h)
        diffused = rearrange(diffused, "s b k ... n d -> b ... n (k s d)")
        return self.dropout(self.mix(torch.cat([x, diffused], dim=-1)))
