"""FEDformer's frequency enhanced blocks with Fourier basis.

Reference: https://github.com/MAZiqing/FEDformer (layers/FourierCorrelation.py).
"""

from typing import ClassVar

import torch
from einops import einsum, rearrange
from torch import Tensor, nn

MODE_SELECTIONS = ("random", "low")
COMPLEX_ACTIVATIONS = ("tanh", "softmax")


def select_modes(seq_len: int, modes: int, mode_select: str) -> Tensor:
    """Sorted indices of `modes` frequencies out of the lowest seq_len // 2: either a random
    subset (drawn from torch's global generator) or the lowest ones."""
    if mode_select not in MODE_SELECTIONS:
        raise ValueError(f"mode_select must be one of {MODE_SELECTIONS}, got {mode_select!r}")
    available = seq_len // 2
    modes = min(modes, available)
    if mode_select == "random":
        return torch.randperm(available)[:modes].sort().values
    return torch.arange(modes)


def complex_activation(scores: Tensor, activation: str) -> Tensor:
    """tanh on real and imaginary part, or softmax over the last axis of the magnitude."""
    if activation == "tanh":
        return torch.complex(scores.real.tanh(), scores.imag.tanh())
    magnitude = torch.softmax(scores.abs(), dim=-1)
    return torch.complex(magnitude, torch.zeros_like(magnitude))


class FourierBlock(nn.Module):
    """Frequency enhanced self-attention: keeps a fixed set of Fourier modes of the queries and
    mixes the channels of each head with a learned complex weight per mode."""

    inputs: ClassVar[tuple[str, ...]] = ("queries",)

    def __init__(self, d_model: int, n_heads: int, seq_len: int, modes: int, mode_select: str):
        super().__init__()
        self.register_buffer("index", select_modes(seq_len, modes, mode_select))
        head_dim = d_model // n_heads
        shape = (n_heads, head_dim, head_dim, len(self.index))
        # uniform in [0, 1 / d_model^2), as in the reference
        scale = 1 / (d_model * d_model)
        self.weight_real = nn.Parameter(scale * torch.rand(shape))
        self.weight_imag = nn.Parameter(scale * torch.rand(shape))

    def forward(self, queries: Tensor) -> Tensor:
        """(b, t, h, e) -> (b, t, h, e)"""
        b, t, h, e = queries.shape
        spectrum = torch.fft.rfft(rearrange(queries, "b t h e -> b h e t"))  # (b, h, e, f)
        weight = torch.complex(self.weight_real, self.weight_imag)
        out = spectrum.new_zeros(b, h, e, t // 2 + 1)
        out[..., self.index] = einsum(
            spectrum[..., self.index], weight, "b h i m, h i o m -> b h o m"
        )
        return rearrange(torch.fft.irfft(out, n=t), "b h e t -> b t h e")


class FourierCrossAttention(nn.Module):
    """Frequency enhanced cross-attention: attention between selected Fourier modes of queries
    and keys (the keys also serve as values, as in the reference), then a learned complex
    channel mixing per query mode.

    The weights start uniform in [0, 1 / d_model^2), as in the reference, which also divides
    the output by d_model^2 a second time. That second division is left out: with it, the
    cross-attention output (the decoder's only path to the encoder) is ~1e-6 of the forecast at
    initialization and stays ~2% after training, so the encoder is all but unused.
    """

    inputs: ClassVar[tuple[str, ...]] = ("queries", "keys")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int,
        mode_select: str,
        activation: str,
    ):
        super().__init__()
        if activation not in COMPLEX_ACTIVATIONS:
            raise ValueError(f"activation must be one of {COMPLEX_ACTIVATIONS}, got {activation!r}")
        self.activation = activation
        self.register_buffer("index_q", select_modes(seq_len_q, modes, mode_select))
        self.register_buffer("index_kv", select_modes(seq_len_kv, modes, mode_select))
        head_dim = d_model // n_heads
        shape = (n_heads, head_dim, head_dim, len(self.index_q))
        # uniform in [0, 1 / d_model^2), as in the reference
        scale = 1 / (d_model * d_model)
        self.weight_real = nn.Parameter(scale * torch.rand(shape))
        self.weight_imag = nn.Parameter(scale * torch.rand(shape))

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """queries (b, t, h, e), keys (b, s, h, e) -> (b, t, h, e)"""
        b, t, h, e = queries.shape
        q = torch.fft.rfft(rearrange(queries, "b t h e -> b h e t"))[..., self.index_q]
        k = torch.fft.rfft(rearrange(keys, "b s h e -> b h e s"))[..., self.index_kv]
        scores = complex_activation(einsum(q, k, "b h e x, b h e y -> b h x y"), self.activation)
        attended = einsum(scores, k, "b h x y, b h e y -> b h e x")
        weight = torch.complex(self.weight_real, self.weight_imag)
        mixed = einsum(attended, weight, "b h e x, h e o x -> b h o x")
        out = q.new_zeros(b, h, e, t // 2 + 1)
        out[..., self.index_q] = mixed
        return rearrange(torch.fft.irfft(out, n=t), "b h e t -> b t h e")
