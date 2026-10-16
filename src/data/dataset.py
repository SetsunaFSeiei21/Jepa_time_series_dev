"""A dataset in the unified on-disk format (see tools/convert_datasets.py), held as tensors.

Shape letters: t=time, n=node (or variable), c=channel, f=time feature, b=batch.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class Batch:
    """A mini-batch of forecasting windows."""

    x: Tensor  # (b, t_in, n, c)
    y: Tensor  # (b, t_out, n, c)
    x_mask: Tensor  # (b, t_in, n, c) bool, True = missing
    y_mask: Tensor  # (b, t_out, n, c) bool
    x_mark: Tensor  # (b, t_in, f), f = 0 when the dataset has no timestamps
    y_mark: Tensor  # (b, t_out, f)

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            **{f.name: getattr(self, f.name).to(device, non_blocking=True) for f in fields(self)}
        )


@dataclass
class SpatioTemporalData:
    name: str
    data: Tensor  # (t, n, c), missing entries are 0
    mask: Tensor  # (t, n, c) bool, True = missing
    time_features: Tensor  # (t, f)
    graph: Tensor | None  # (n, n) static adjacency
    meta: dict

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> "SpatioTemporalData":
        path = Path(path).expanduser()
        with open(path / "meta.json") as f:
            meta = json.load(f)

        def load_array(name: str) -> Tensor | None:
            file = path / f"{name}.npy"
            return torch.from_numpy(np.load(file)).to(device) if file.exists() else None

        time_features = load_array("time_features")
        if time_features is None:
            time_features = torch.zeros(meta["num_steps"], 0, device=device)
        return cls(
            name=meta["name"],
            data=load_array("data"),
            mask=load_array("mask"),
            time_features=time_features,
            graph=load_array("graph"),
            meta=meta,
        )

    @property
    def num_steps(self) -> int:
        return self.data.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.data.shape[1]

    @property
    def num_channels(self) -> int:
        return self.data.shape[2]

    @property
    def num_time_features(self) -> int:
        return self.time_features.shape[1]

    @property
    def time_feature_sizes(self) -> dict[str, int]:
        """Feature name -> number of distinct values, in column order."""
        return self.meta["time_features"]

    def split_bounds(self, ratios: list[float]) -> list[tuple[int, int]]:
        """[lo, hi) step ranges of the train/val/test splits; test takes the remainder."""
        train_ratio, val_ratio, test_ratio = ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6, ratios
        train_end = int(self.num_steps * train_ratio)
        val_end = train_end + int(self.num_steps * val_ratio)
        return [(0, train_end), (train_end, val_end), (val_end, self.num_steps)]

    def window_starts(self, lo: int, hi: int, his_len: int, pred_len: int, stride: int) -> Tensor:
        """Start steps of all windows whose target lies inside [lo, hi).

        The input part of a window may reach back before `lo`, so the first
        val/test windows read history from the preceding split.
        """
        first = max(lo - his_len, 0)
        last = hi - his_len - pred_len
        return torch.arange(first, last + 1, stride, device=self.data.device)

    def windows(self, starts: Tensor, his_len: int, pred_len: int) -> Batch:
        steps = starts[:, None] + torch.arange(his_len + pred_len, device=starts.device)
        data, mask, mark = self.data[steps], self.mask[steps], self.time_features[steps]
        return Batch(
            x=data[:, :his_len],
            y=data[:, his_len:],
            x_mask=mask[:, :his_len],
            y_mask=mask[:, his_len:],
            x_mark=mark[:, :his_len],
            y_mark=mark[:, his_len:],
        )
