from dataclasses import dataclass

import torch
from torch import Tensor

# Standard deviations below this are treated as a constant series and left unscaled.
MIN_STD = 1e-5

# Reduction dims over a (t, n, c) series for dataset-level normalization.
DATASET_NORM_DIMS = {
    "node": (0,),  # statistics per node and channel
    "global": (0, 1),  # statistics per channel, shared by all nodes
}


@dataclass
class StandardScaler:
    """Z-score scaler whose statistics broadcast against the data they were fitted on."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x: Tensor, mask: Tensor, dims: tuple[int, ...]) -> "StandardScaler":
        """Masked mean/std of `x` over `dims` (kept as size-1); `mask` is True where missing."""
        valid = (~mask).to(x.dtype)
        count = valid.sum(dims, keepdim=True).clamp(min=1)
        mean = (x * valid).sum(dims, keepdim=True) / count
        var = ((x - mean).square() * valid).sum(dims, keepdim=True) / count
        std = var.sqrt()
        return cls(mean, torch.where(std < MIN_STD, torch.ones_like(std), std))

    @classmethod
    def identity(cls, device: torch.device) -> "StandardScaler":
        return cls(torch.zeros((), device=device), torch.ones((), device=device))

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def to(self, device: torch.device) -> "StandardScaler":
        return StandardScaler(self.mean.to(device), self.std.to(device))


def fit_dataset_scaler(data: Tensor, mask: Tensor, train_end: int, mode: str) -> StandardScaler:
    """Fit on the training span of a (t, n, c) series; mode is node | global | none."""
    if mode == "none":
        return StandardScaler.identity(data.device)
    train = data[:train_end].double()
    scaler = StandardScaler.fit(train, mask[:train_end], DATASET_NORM_DIMS[mode])
    return StandardScaler(scaler.mean.to(data.dtype), scaler.std.to(data.dtype))
