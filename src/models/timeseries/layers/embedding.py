import math

import torch
from einops import rearrange
from torch import Tensor, nn

# Wavelength base of the sinusoidal position encoding (Vaswani et al., 2017).
POSITION_BASE = 10000.0


def sinusoidal_positions(length: int, d_model: int) -> Tensor:
    """(length, d_model): sin on even, cos on odd channels."""
    position = rearrange(torch.arange(length, dtype=torch.float), "t -> t 1")
    frequency = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(POSITION_BASE) / d_model)
    )
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding


class DataEmbedding(nn.Module):
    """Token embedding of the Autoformer family: a circular convolution over time of all
    variates, plus optional sinusoidal positions and a linear embedding of the time features
    (TSLib's `timeF`; omitted without time features), followed by dropout."""

    def __init__(
        self,
        num_variates: int,
        num_time_features: int,
        d_model: int,
        kernel_size: int,
        max_len: int,
        positional: bool,
        dropout: float,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"kernel_size must be odd to keep the length, got {kernel_size}"
        )
        self.value_embedding = nn.Conv1d(
            num_variates,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.value_embedding.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        positions = sinusoidal_positions(max_len, d_model) if positional else None
        self.register_buffer("positions", positions, persistent=False)
        self.time_embedding = (
            nn.Linear(num_time_features, d_model, bias=False) if num_time_features > 0 else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """x (b, t, v), x_mark (b, t, f) -> (b, t, d)"""
        h = rearrange(self.value_embedding(rearrange(x, "b t v -> b v t")), "b d t -> b t d")
        if self.positions is not None:
            h = h + self.positions[: x.shape[1]]
        if self.time_embedding is not None:
            h = h + self.time_embedding(x_mark)
        return self.dropout(h)
