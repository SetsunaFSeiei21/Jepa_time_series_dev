# src/dataset/presplit_mts.py

import json
from os.path import exists, join
from typing import Optional

import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset


class PreSplitMTSDataset(Dataset):
    """
    Dataset for pre-split multivariate time-series data.

    Expected directory format:

        dataset_path/
        ├── meta.json
        ├── train_data.npy
        ├── train_timestamps.npy        # optional
        ├── val_data.npy
        ├── val_timestamps.npy          # optional
        ├── test_data.npy
        └── test_timestamps.npy         # optional

    This dataset follows Scheme A:
        - train split only reads train_data.npy
        - val split only reads val_data.npy
        - test split only reads test_data.npy

    It does NOT borrow history from the previous split.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        data_path,
        his_len: int,
        pred_len: int,
        split: str,
        timestamp_dim: int = 4,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split={split}. Expected one of {self.VALID_SPLITS}."
            )

        self.data_path = data_path
        self.his_len = his_len
        self.pred_len = pred_len
        self.seq_len = his_len + pred_len
        self.split = split
        self.timestamp_dim = timestamp_dim

        meta_path = join(data_path, "meta.json")
        if not exists(meta_path):
            raise FileNotFoundError(f"meta.json not found at {meta_path}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        data_file = join(data_path, f"{split}_data.npy")
        if not exists(data_file):
            raise FileNotFoundError(f"{split}_data.npy not found at {data_file}")

        data = np.load(data_file)

        if data.ndim == 1:
            data = data[:, None]

        if data.ndim != 2:
            raise ValueError(
                f"{split}_data.npy should have shape (T, N), "
                f"but got shape {data.shape}."
            )

        self.data = data.astype(np.float32)
        self.data_len, self.num_nodes = self.data.shape

        # null value handling
        regular_settings = self.meta.get("regular_settings", {})
        null_val = regular_settings.get("null_val", np.nan)

        if null_val is None:
            nan_mask = np.zeros_like(self.data, dtype=bool)
        elif isinstance(null_val, float) and np.isnan(null_val):
            nan_mask = np.isnan(self.data)
        else:
            nan_mask = self.data == null_val

        # Optional split-level mask.
        # Your uploaded time-series data does not seem to include these files,
        # but this keeps the class extensible.
        mask_file = join(data_path, f"{split}_mask.npy")
        if exists(mask_file):
            extra_mask = np.load(mask_file).astype(bool)
            if extra_mask.shape != self.data.shape:
                raise ValueError(
                    f"{split}_mask.npy shape {extra_mask.shape} does not match "
                    f"{split}_data.npy shape {self.data.shape}."
                )
            self.mask = nan_mask | extra_mask
        else:
            self.mask = nan_mask

        # Timestamp features.
        #
        # Most uploaded datasets have:
        #   train_timestamps.npy / val_timestamps.npy / test_timestamps.npy
        # with shape (T, 4).
        #
        # Pulse has no timestamps, so we create zero time features.
        timestamp_file = join(data_path, f"{split}_timestamps.npy")
        if exists(timestamp_file):
            timestamp = np.load(timestamp_file).astype(np.float32)

            if timestamp.ndim == 1:
                timestamp = timestamp[:, None]

            if timestamp.shape[0] != self.data_len:
                raise ValueError(
                    f"{split}_timestamps.npy length {timestamp.shape[0]} does not "
                    f"match {split}_data.npy length {self.data_len}."
                )

            self.timestamp = timestamp
        else:
            # For datasets like Pulse.
            # Use 4 zero features because Autoformer/FEDformer timeF embedding with
            # freq='h' expects 4 time features.
            self.timestamp = np.zeros(
                (self.data_len, timestamp_dim),
                dtype=np.float32,
            )

        self.num_features = self.timestamp.shape[1]

        if self.data_len < self.seq_len:
            raise ValueError(
                f"{split} split is too short: data_len={self.data_len}, "
                f"but his_len + pred_len = {self.seq_len}."
            )

    def __len__(self):
        return self.data_len - self.seq_len + 1

    def cal_stat(self):
        """
        Compute statistics on this split.

        For training normalization, call this only on train_dataset.
        """
        x_mean = np.nanmean(self.data, axis=0, keepdims=True)
        x_std = np.nanstd(self.data, axis=0, keepdims=True)

        return (
            repeat(torch.from_numpy(x_mean), "1 n -> 1 1 n 1"),
            repeat(torch.from_numpy(x_std), "1 n -> 1 1 n 1"),
            self.num_features,
        )

    def __getitem__(self, idx):
        """
        Return:
            input_seq:       (his_len, num_nodes, 1)
            target_seq:      (pred_len, num_nodes, 1)
            input_features:  (his_len, num_features)
            target_features: (pred_len, num_features)
            input_mask:      (his_len, num_nodes, 1)
            target_mask:     (pred_len, num_nodes, 1)
        """
        input_start = idx
        input_end = idx + self.his_len
        target_end = idx + self.seq_len

        input_seq = (
            torch.from_numpy(self.data[input_start:input_end].copy())
            .unsqueeze(-1)
            .float()
        )

        target_seq = (
            torch.from_numpy(self.data[input_end:target_end].copy())
            .unsqueeze(-1)
            .float()
        )

        input_features = torch.from_numpy(
            self.timestamp[input_start:input_end].copy()
        ).float()

        target_features = torch.from_numpy(
            self.timestamp[input_end:target_end].copy()
        ).float()

        input_mask = (
            torch.from_numpy(self.mask[input_start:input_end].copy())
            .unsqueeze(-1)
            .bool()
        )

        target_mask = (
            torch.from_numpy(self.mask[input_end:target_end].copy())
            .unsqueeze(-1)
            .bool()
        )

        return (
            input_seq,
            target_seq,
            input_features,
            target_features,
            input_mask,
            target_mask,
        )