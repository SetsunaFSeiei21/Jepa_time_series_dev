"""FEDformer's frequency enhanced blocks with a Legendre multiwavelet basis.

Reference: https://github.com/MAZiqing/FEDformer (layers/MultiWaveletCorrelation.py), built on
the multiwavelet neural operator of Gupta et al., https://arxiv.org/abs/2109.13459.

Shape letters: `c` wavelet channels, `k` polynomial order of the basis.
"""

import math
from typing import ClassVar

import numpy as np
import torch
from einops import einsum, rearrange
from numpy.polynomial import legendre
from torch import Tensor, nn

from .attention import fit_length
from .fourier import COMPLEX_ACTIVATIONS, complex_activation

# Gram-Schmidt passes per vector; a second pass restores orthogonality to rounding level.
GRAM_SCHMIDT_PASSES = 2


def legendre_filters(k: int) -> dict[str, Tensor]:
    """Two-scale filters of the order-k Legendre multiwavelets (Alpert), as (2k, k) matrices:
    `decompose_scaling` / `decompose_detail` map an [even, odd] pair of fine coefficients to the
    coarse scaling / detail coefficients, `reconstruct_even` / `reconstruct_odd` map
    [scaling, detail] back to the even / odd fine coefficients.

    Everything is computed in the orthonormal basis of the fine scale, sqrt(2) phi_i(2x) on
    [0, 1/2] and sqrt(2) phi_i(2x - 1) on [1/2, 1], with phi_i(x) = sqrt(2i + 1) P_i(2x - 1):
    the coarse scaling functions are exact Gauss-Legendre projections onto it, and the wavelets
    are the left halves Gram-Schmidt orthogonalized against them and each other (the same
    construction as the reference, which works on monomial coefficients and loses all
    precision at k = 8). The filter bank [H0 H1; G0 G1] is orthogonal to rounding level.
    """
    nodes, weights = legendre.leggauss(k)
    x, w = (nodes + 1) / 2, weights / 2  # Gauss-Legendre quadrature on [0, 1]
    normalization = np.diag(np.sqrt(2 * np.arange(k) + 1))

    def phi(points):
        return legendre.legval(2 * points - 1, normalization)  # (k, points)

    # <phi_i, fine basis function j> on each half, exact for these polynomial degrees
    h0 = phi(x / 2) * w @ phi(x).T / math.sqrt(2)
    h1 = phi((x + 1) / 2) * w @ phi(x).T / math.sqrt(2)
    basis = [*np.concatenate([h0, h1], axis=1)]  # orthonormal rows, in R^2k
    for i in range(k):
        v = np.eye(2 * k)[i]
        for _ in range(GRAM_SCHMIDT_PASSES):
            v = v - np.stack(basis).T @ (np.stack(basis) @ v)
        basis.append(v / np.linalg.norm(v))
    g0, g1 = np.split(np.stack(basis[k:]), 2, axis=1)
    filters = {
        "decompose_scaling": np.concatenate([h0.T, h1.T]),
        "decompose_detail": np.concatenate([g0.T, g1.T]),
        "reconstruct_even": np.concatenate([h0, g0]),
        "reconstruct_odd": np.concatenate([h1, g1]),
    }
    return {name: torch.tensor(f, dtype=torch.float) for name, f in filters.items()}


class _WaveletFilters(nn.Module):
    """One level of the multiwavelet transform and its inverse along time."""

    def __init__(self, k: int):
        super().__init__()
        for name, matrix in legendre_filters(k).items():
            self.register_buffer(name, matrix, persistent=False)

    def decompose(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(b, t, c, k) -> detail, scaling, both (b, t / 2, c, k)"""
        pairs = rearrange(x, "b (t two) c k -> b t c (two k)", two=2)
        detail = einsum(pairs, self.decompose_detail, "b t c j, j k -> b t c k")
        return detail, einsum(pairs, self.decompose_scaling, "b t c j, j k -> b t c k")

    def reconstruct(self, x: Tensor) -> Tensor:
        """[scaling, detail] (b, t, c, 2k) -> (b, 2t, c, k)"""
        even = einsum(x, self.reconstruct_even, "b t c j, j k -> b t c k")
        odd = einsum(x, self.reconstruct_odd, "b t c j, j k -> b t c k")
        return rearrange([even, odd], "two b t c k -> b (t two) c k")


def _pad_to_power_of_two(x: Tensor) -> tuple[Tensor, int]:
    """Appends the first steps again until the length is a power of two: (b, t, ...) ->
    (b, 2^ceil(log2 t), ...), and returns the number of decomposition levels floor(log2 t)."""
    t = x.shape[1]
    padded = 2 ** math.ceil(math.log2(t))
    return torch.cat([x, x[:, : padded - t]], dim=1), math.floor(math.log2(t))


class _SparseFourierKernel(nn.Module):
    """Learned complex mixing of all c * k channels on the lowest `modes` Fourier modes."""

    def __init__(self, k: int, modes: int, channels: int):
        super().__init__()
        self.modes = modes
        width = channels * k
        shape = (width, width, modes)
        scale = 1 / (width * width)  # uniform in [0, 1 / width^2), as in the reference
        self.weight_real = nn.Parameter(scale * torch.rand(shape))
        self.weight_imag = nn.Parameter(scale * torch.rand(shape))

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, c, k) -> (b, t, c, k)"""
        b, t, c, _ = x.shape
        spectrum = torch.fft.rfft(rearrange(x, "b t c k -> b (c k) t"))
        kept = min(self.modes, t // 2 + 1)
        weight = torch.complex(self.weight_real[..., :kept], self.weight_imag[..., :kept])
        out = spectrum.new_zeros(spectrum.shape)
        out[..., :kept] = einsum(spectrum[..., :kept], weight, "b i m, i o m -> b o m")
        return rearrange(torch.fft.irfft(out, n=t), "b (c k) t -> b t c k", c=c)


class _MultiwaveletKernel(nn.Module):
    """Multiwavelet neural operator: decomposes the sequence level by level, applies the
    sparse Fourier kernels A, B, C (named as in the paper) to the detail / scaling coefficients
    of every level and a linear map to the coarsest scaling coefficients, and reconstructs."""

    def __init__(self, k: int, modes: int, channels: int, skip_levels: int):
        super().__init__()
        self.skip_levels = skip_levels
        self.filters = _WaveletFilters(k)
        self.kernel_a = _SparseFourierKernel(k, modes, channels)
        self.kernel_b = _SparseFourierKernel(k, modes, channels)
        self.kernel_c = _SparseFourierKernel(k, modes, channels)
        self.coarse_transform = nn.Linear(k, k)

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, c, k) -> (b, t, c, k)"""
        t = x.shape[1]
        x, levels = _pad_to_power_of_two(x)
        details, scalings = [], []
        for _ in range(levels - self.skip_levels):
            detail, x = self.filters.decompose(x)
            details.append(self.kernel_a(detail) + self.kernel_b(x))
            scalings.append(self.kernel_c(detail))
        x = self.coarse_transform(x)
        for detail, scaling in zip(reversed(details), reversed(scalings), strict=True):
            x = self.filters.reconstruct(torch.cat([x + scaling, detail], dim=-1))
        return x[:, :t]


class MultiWaveletTransform(nn.Module):
    """Frequency enhanced self-attention with multiwavelets: lifts the values to c * k wavelet
    channels and applies `num_blocks` multiwavelet kernels (ReLU in between)."""

    inputs: ClassVar[tuple[str, ...]] = ("values",)

    def __init__(
        self, d_model: int, k: int, modes: int, channels: int, num_blocks: int, skip_levels: int
    ):
        super().__init__()
        self.channels = channels
        self.lift = nn.Linear(d_model, channels * k)
        self.blocks = nn.ModuleList(
            _MultiwaveletKernel(k, modes, channels, skip_levels) for _ in range(num_blocks)
        )
        self.project = nn.Linear(channels * k, d_model)

    def forward(self, values: Tensor) -> Tensor:
        """(b, t, h, e) -> (b, t, h, e)"""
        heads = values.shape[2]
        x = self.lift(rearrange(values, "b t h e -> b t (h e)"))
        x = rearrange(x, "b t (c k) -> b t c k", c=self.channels)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = torch.relu(x)
        x = self.project(rearrange(x, "b t c k -> b t (c k)"))
        return rearrange(x, "b t (h e) -> b t h e", h=heads)


class MultiWaveletCross(nn.Module):
    """Frequency enhanced cross-attention with multiwavelets: decomposes lifted queries and
    keys level by level and runs a Fourier cross-attention on the lowest `modes` modes of every
    level's coefficients (the keys also serve as values, as in the reference)."""

    inputs: ClassVar[tuple[str, ...]] = ("queries", "keys")

    def __init__(
        self, d_model: int, k: int, channels: int, modes: int, skip_levels: int, activation: str
    ):
        super().__init__()
        if activation not in COMPLEX_ACTIVATIONS:
            raise ValueError(f"activation must be one of {COMPLEX_ACTIVATIONS}, got {activation!r}")
        self.channels = channels
        self.modes = modes
        self.skip_levels = skip_levels
        self.activation = activation
        self.scale = 1 / (d_model * d_model)
        self.filters = _WaveletFilters(k)
        self.lift_queries = nn.Linear(d_model, channels * k)
        self.lift_keys = nn.Linear(d_model, channels * k)
        self.project = nn.Linear(channels * k, d_model)

    def _attend(self, queries: Tensor, keys: Tensor) -> Tensor:
        """Fourier cross-attention over time, heads = polynomial order: (b, t, c, k) -> same"""
        t, s = queries.shape[1], keys.shape[1]
        q = torch.fft.rfft(rearrange(queries, "b t c k -> b k c t"))[..., : min(t // 2, self.modes)]
        kv = torch.fft.rfft(rearrange(keys, "b s c k -> b k c s"))[..., : min(s // 2, self.modes)]
        scores = complex_activation(einsum(q, kv, "b k c x, b k c y -> b k x y"), self.activation)
        attended = einsum(scores, kv, "b k x y, b k c y -> b k c x")
        out = q.new_zeros(*q.shape[:-1], t // 2 + 1)
        out[..., : q.shape[-1]] = attended
        return rearrange(torch.fft.irfft(out * self.scale, n=t), "b k c t -> b t c k")

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """queries (b, t, h, e), keys (b, s, h, e) -> (b, t, h, e)"""
        t, heads = queries.shape[1], queries.shape[2]
        q, k = (
            rearrange(
                lift(rearrange(x, "b t h e -> b t (h e)")), "b t (c k) -> b t c k", c=self.channels
            )
            for lift, x in ((self.lift_queries, queries), (self.lift_keys, keys))
        )
        q, levels = _pad_to_power_of_two(q)
        k, _ = _pad_to_power_of_two(fit_length(k, t))
        details, scalings = [], []
        for _ in range(levels - self.skip_levels):
            detail_q, q = self.filters.decompose(q)
            detail_k, k = self.filters.decompose(k)
            attended_detail = self._attend(detail_q, detail_k)
            details.append(attended_detail + self._attend(q, k))
            # the reference also feeds the detail coefficients (not the scaling ones) here
            scalings.append(attended_detail)
        x = self._attend(q, k)
        for detail, scaling in zip(reversed(details), reversed(scalings), strict=True):
            x = self.filters.reconstruct(torch.cat([x + scaling, detail], dim=-1))
        x = self.project(rearrange(x[:, :t], "b t c k -> b t (c k)"))
        return rearrange(x, "b t (h e) -> b t h e", h=heads)
