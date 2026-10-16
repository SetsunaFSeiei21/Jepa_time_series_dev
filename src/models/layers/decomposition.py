import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn


class MovingAverage(nn.Module):
    """Centered moving average over time; series ends are padded by repeating the edge value."""

    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"kernel_size must be odd to keep the length, got {kernel_size}"
        )
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """(b, t, n, c) -> (b, t, n, c)"""
        half = self.kernel_size // 2
        front = repeat(x[:, :1], "b 1 n c -> b k n c", k=half)
        end = repeat(x[:, -1:], "b 1 n c -> b k n c", k=half)
        padded = rearrange(torch.cat([front, x, end], dim=1), "b t n c -> (b n) c t")
        smoothed = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        return rearrange(smoothed, "(b n) c t -> b t n c", n=x.shape[2])


class SeriesDecomposition(nn.Module):
    """Splits a series into a seasonal remainder and a moving-average trend (Autoformer)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(b, t, n, c) -> seasonal, trend, both (b, t, n, c)"""
        trend = self.moving_average(x)
        return x - trend, trend
