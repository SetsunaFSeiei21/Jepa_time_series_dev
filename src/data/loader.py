import math
from collections.abc import Iterator

import torch
from torch import Tensor

from .dataset import Batch, SpatioTemporalData


class WindowLoader:
    """Iterates over sliding windows of a dataset by gathering them directly from its tensors.

    The whole series stays on its storage device; batches are moved to `device`.
    """

    def __init__(
        self,
        dataset: SpatioTemporalData,
        starts: Tensor,
        his_len: int,
        pred_len: int,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
    ):
        self.dataset = dataset
        self.starts = starts
        self.his_len = his_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __len__(self) -> int:
        return math.ceil(len(self.starts) / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        order = (
            torch.randperm(len(self.starts), device=self.starts.device)
            if self.shuffle
            else torch.arange(len(self.starts), device=self.starts.device)
        )
        for chunk in self.starts[order].split(self.batch_size):
            yield self.dataset.windows(chunk, self.his_len, self.pred_len).to(self.device)


def build_loaders(
    dataset: SpatioTemporalData,
    split: list[float],
    his_len: int,
    pred_len: int,
    train_stride: int,
    eval_stride: int,
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, WindowLoader]:
    """Train windows slide by `train_stride`; val/test windows by `eval_stride`."""
    loaders = {}
    for name, (lo, hi) in zip(("train", "val", "test"), dataset.split_bounds(split), strict=True):
        is_train = name == "train"
        starts = dataset.window_starts(
            lo, hi, his_len, pred_len, train_stride if is_train else eval_stride
        )
        loaders[name] = WindowLoader(
            dataset,
            starts,
            his_len,
            pred_len,
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=is_train,
            device=device,
        )
    return loaders
