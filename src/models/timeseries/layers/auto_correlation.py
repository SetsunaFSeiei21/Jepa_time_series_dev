import math
from typing import ClassVar

import torch
from einops import einsum, rearrange, reduce, repeat
from torch import Tensor, nn

from .attention import fit_length


class AutoCorrelation(nn.Module):
    """Autoformer's auto-correlation: finds the `factor * ln(t)` most correlated time delays via
    FFT and aggregates the values rolled by those delays, weighted by the softmax of their
    correlation.

    The delays and their weights are chosen per sample (from the correlation averaged over heads
    and channels) in training and inference alike, so a sample's output never depends on the
    other samples of its batch. The reference implementation shares one set of delays across
    the batch during training as a speed-up, which makes the training-mode function depend on
    the batch composition and differ from the inference-mode one.
    """

    inputs: ClassVar[tuple[str, ...]] = ("queries", "keys", "values")

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        """queries (b, t, h, e), keys / values (b, s, h, e) -> (b, t, h, e)"""
        length = queries.shape[1]
        q, k, v = (
            rearrange(x, "b t h e -> b h e t")
            for x in (queries, fit_length(keys, length), fit_length(values, length))
        )
        # circular cross-correlation of queries and keys for every delay: (b, h, e, t)
        corr = torch.fft.irfft(torch.fft.rfft(q) * torch.fft.rfft(k).conj(), n=length)
        return rearrange(self._aggregate(v, corr), "b h e t -> b t h e")

    def _aggregate(self, values: Tensor, corr: Tensor) -> Tensor:
        """values, corr: (b, h, e, t) -> (b, h, e, t)"""
        _, heads, channels, length = values.shape
        top_k = int(self.factor * math.log(length))
        mean_corr = reduce(corr, "b h e t -> b t", "mean")
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)  # (b, k)
        weights = torch.softmax(weights, dim=-1)  # (b, k)
        # rolled[..., k, t] = values[..., (t + delay_k) mod length]
        index = (delays[..., None] + torch.arange(length, device=values.device)) % length
        index = repeat(index, "b k t -> b h e k t", h=heads, e=channels)
        rolled = torch.gather(repeat(values, "b h e t -> b h e k t", k=top_k), -1, index)
        return einsum(rolled, weights, "b h e k t, b k -> b h e t")
