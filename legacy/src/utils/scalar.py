from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def nanstd(tensor, dim=None, keepdim=False, correction=1):
    count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=keepdim)
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    sq_diff = (tensor - mean).pow(2)
    sq_diff = torch.where(
        torch.isnan(sq_diff), torch.tensor(0.0, device=tensor.device), sq_diff
    )
    sum_sq_diff = sq_diff.sum(dim=dim, keepdim=keepdim)
    divisor = count - correction
    divisor = torch.clamp(divisor, min=1)
    return (sum_sq_diff / divisor).sqrt()


class Normalizer(ABC):
    """Abstract base class for data normalization."""

    @abstractmethod
    def fit(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Fit normalizer to data."""
        pass

    @abstractmethod
    def transform(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform data using fitted parameters."""
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data back to original scale."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get normalization parameters."""
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Set normalization parameters."""
        pass


class ZScoreNormalizer(Normalizer):
    """
    Z-score normalizer with mask support.

    Formula: (x - mean) / std
    Handles masked values by ignoring them in mean/std computation.
    """

    def __init__(self, x_mean=None, x_std=None):
        self.x_mean = x_mean
        self.x_std = x_std

    def fit(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Compute mean and std from data, optionally ignoring masked values.

        Args:
            data: Tensor of shape (B, T, N, F)
            mask: Boolean tensor of same shape as data (True = ignore)
        """
        if mask is not None:
            data[mask] = torch.nan

        self.x_mean = data.nanmean(dim=1, keepdim=True)
        self.x_std = nanstd(data, dim=1, keepdim=True)

        self.x_mean = torch.nan_to_num(self.x_mean, nan=0)
        self.x_std = torch.nan_to_num(self.x_std, nan=1)
        self.x_std = self.x_std.clamp(min=1)

        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization.

        Args:
            data: Tensor of shape (B, T, N, F)
            mask: Boolean tensor of same shape as data (True = ignore)
        """
        if self.x_mean is None or self.x_std is None:
            raise ValueError("Normalizer must be fitted before transform")

        return (data - self.x_mean) / self.x_std

    def fit_transform(self, data: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.fit(data, mask).transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse z-score normalization."""
        if self.x_mean is None or self.x_std is None:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        return data * self.x_std + self.x_mean

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get normalization parameters."""
        return {"mean": self.x_mean.clone(), "std": self.x_std.clone()}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Set normalization parameters."""
        self.x_mean = params["mean"]
        self.x_std = params["std"]


class NormalizationPipeline:
    def __init__(
        self,
        dataset_normalizer: Optional[ZScoreNormalizer] = None,
        batch_normalizer: Optional[ZScoreNormalizer] = None,
        strategy="tuning",
    ):
        self.dataset_normalizer = dataset_normalizer
        self.batch_normalizer = batch_normalizer

        assert strategy in ["pretrain", "tuning"]
        self.strategy = strategy

    def fit(self, data):
        self.batch_normalizer.fit(data)

    def fit_transform(self, x):
        if self.strategy == "tuning":
            if self.dataset_normalizer:
                x = self.dataset_normalizer.transform(x)

        if self.batch_normalizer:
            x = self.batch_normalizer.fit_transform(x)

        return x

    def transform(self, y):
        if self.strategy == "tuning":
            if self.dataset_normalizer:
                y = self.dataset_normalizer.transform(y)

        if self.batch_normalizer:
            y = self.batch_normalizer.transform(y)

        return y

    def inverse_transform(self, y):
        if self.batch_normalizer:
            y = self.batch_normalizer.inverse_transform(y)

        if self.strategy == "tuning":
            if self.dataset_normalizer:
                y = self.dataset_normalizer.inverse_transform(y)

        return y
