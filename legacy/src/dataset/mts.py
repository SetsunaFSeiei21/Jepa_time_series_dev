import json
from os.path import exists, join
from typing import Union

import numpy as np
import pandas as pd
import torch
from einops import repeat
from torch.utils.data import Dataset

from .time_features import time_features


class MTSDataset(Dataset):

    def __init__(self, data_path, his_len, pred_len):
        data = np.load(join(data_path, "data.npy"))

        with open(join(data_path, "dataset_config.json"), "r") as f:
            dataset_config = json.load(f)
        nan_value = dataset_config["null_val"]
        nan_mask = data == nan_value
        if exists(join(data_path, "mask.npy")):
            mask = np.load(join(data_path, "mask.npy"))
        self.mask = nan_mask | mask

        timestamp = pd.date_range(
            start=dataset_config["start"],
            end=dataset_config["end"],
            freq=dataset_config["freq"],
        ).to_series()

        self.data = data
        self.timestamp = time_features(
            timestamp.dt,
            timestamp.dt.freq,
        ).T

        self.data_len, self.num_nodes = data.shape
        self.num_features = self.timestamp.shape[1]
        self.his_len = his_len
        self.pred_len = pred_len
        self.seq_len = his_len + pred_len
        self.is_norm = False

    def __len__(self):
        return self.data_len - self.seq_len + 1

    def cal_stat(self, train_point, train_val_point):
        train_data = self.data[train_point:train_val_point]
        x_mean = np.nanmean(train_data, axis=0, keepdims=True)
        x_std = np.nanstd(train_data, axis=0, keepdims=True)
        return (
            repeat(torch.from_numpy(x_mean),"1 n -> 1 1 n 1"),
            repeat(torch.from_numpy(x_std),"1 n -> 1 1 n 1"),
            self.num_features,
        )

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Returns:
            Tuple containing:
            - input_seq: (his_len, num_nodes, 1)
            - target_seq: (pred_len, num_nodes, 1)
            - input_features: (his_len, num_features)
            - target_features: (pred_len, num_features)
            - input_mask: (his_len, num_nodes, 1)
            - target_mask: (pred_len, num_nodes, 1)
        """
        # Convert numpy arrays to tensors
        input_seq = (
            torch.from_numpy(self.data[idx : idx + self.his_len].copy())
            .unsqueeze(-1)
            .float()
        )
        target_seq = (
            torch.from_numpy(self.data[idx + self.his_len : idx + self.seq_len].copy())
            .unsqueeze(-1)
            .float()
        )

        input_features = torch.from_numpy(
            self.timestamp[idx : idx + self.his_len].copy()
        ).float()
        target_features = torch.from_numpy(
            self.timestamp[idx + self.his_len : idx + self.seq_len].copy()
        ).float()

        input_mask = (
            torch.from_numpy(self.mask[idx : idx + self.his_len].copy())
            .unsqueeze(-1)
            .bool()
        )
        target_mask = (
            torch.from_numpy(self.mask[idx + self.his_len : idx + self.seq_len].copy())
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
