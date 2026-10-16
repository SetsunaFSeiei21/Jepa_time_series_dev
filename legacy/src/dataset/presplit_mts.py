# src/dataset/presplit_mts.py

import json
from os.path import exists, join
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset


class PreSplitMTSDataset(Dataset):
    """
    Dataset for pre-split multivariate time-series data.

    Reference modes:

    1. none
       Ordinary forecasting / JEPA behavior.

    2. pretrain
       For every reference ratio r:
           C_r = train_data[:floor(r * T_train)]

       Train windows using C_r start after C_r.

    3. finetune
       For every checkpoint ratio r:
           C_r = train_data[:floor(r * T_train)]

       Full train split is still used for finetuning.
    """

    VALID_SPLITS = {"train", "val", "test"}
    VALID_REFERENCE_MODES = {
        "none",
        "pretrain",
        "finetune",
    }

    def __init__(
        self,
        data_path,
        his_len: int,
        pred_len: int,
        split: str,
        timestamp_dim: int = 4,
        reference_ratios: Optional[
            Sequence[float]
        ] = None,
        reference_mode: str = "none",
        append_relative_time: bool = False,
        global_time_offset: int = 0,
        train_data_len: Optional[int] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split={split}. "
                f"Expected one of {self.VALID_SPLITS}."
            )

        reference_mode = str(
            reference_mode
        ).lower()

        if (
            reference_mode
            not in self.VALID_REFERENCE_MODES
        ):
            raise ValueError(
                "reference_mode must be one of "
                f"{self.VALID_REFERENCE_MODES}, "
                f"but got {reference_mode}."
            )

        self.data_path = data_path
        self.his_len = int(his_len)
        self.pred_len = int(pred_len)
        self.seq_len = (
            self.his_len + self.pred_len
        )
        self.split = split
        self.timestamp_dim = int(
            timestamp_dim
        )

        self.reference_mode = (
            reference_mode
        )

        self.append_relative_time = bool(
            append_relative_time
        )

        self.global_time_offset = int(
            global_time_offset
        )

        # ------------------------------------------------------------
        # Reference ratios
        # ------------------------------------------------------------

        if reference_ratios is None:
            reference_ratios = []

        self.reference_ratios = [
            float(ratio)
            for ratio in reference_ratios
        ]

        if self.reference_mode == "none":
            if len(self.reference_ratios) != 0:
                raise ValueError(
                    "reference_ratios must be empty "
                    "when reference_mode='none'."
                )

            if self.append_relative_time:
                raise ValueError(
                    "append_relative_time must be false "
                    "when reference_mode='none'."
                )

        else:
            if len(self.reference_ratios) == 0:
                raise ValueError(
                    "reference_ratios must not be empty "
                    "when reference mode is enabled."
                )

            if (
                len(set(self.reference_ratios))
                != len(self.reference_ratios)
            ):
                raise ValueError(
                    "reference_ratios contains duplicate "
                    f"values: {self.reference_ratios}."
                )

            for ratio in self.reference_ratios:
                if self.reference_mode == "pretrain":
                    valid = 0.0 < ratio < 1.0
                else:
                    valid = 0.0 < ratio <= 1.0

                if not valid:
                    raise ValueError(
                        "Invalid reference ratio: "
                        f"mode={self.reference_mode}, "
                        f"ratio={ratio}."
                    )

        self.train_data_len = (
            int(train_data_len)
            if train_data_len is not None
            else None
        )

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        meta_path = join(
            data_path,
            "meta.json",
        )

        if not exists(meta_path):
            raise FileNotFoundError(
                f"meta.json not found at {meta_path}"
            )

        with open(meta_path, "r") as file:
            self.meta = json.load(file)

        # ------------------------------------------------------------
        # Main data
        # ------------------------------------------------------------

        data_file = join(
            data_path,
            f"{split}_data.npy",
        )

        if not exists(data_file):
            raise FileNotFoundError(
                f"{split}_data.npy not found "
                f"at {data_file}"
            )

        data = np.load(data_file)

        if data.ndim == 1:
            data = data[:, None]

        if data.ndim != 2:
            raise ValueError(
                f"{split}_data.npy should have "
                f"shape (T,N), but got {data.shape}."
            )

        self.data = data.astype(
            np.float32
        )

        (
            self.data_len,
            self.num_nodes,
        ) = self.data.shape

        if self.data_len < self.seq_len:
            raise ValueError(
                f"{split} split is too short: "
                f"data_len={self.data_len}, "
                f"his_len+pred_len={self.seq_len}."
            )

        if self.reference_mode != "none":
            if self.train_data_len is None:
                raise ValueError(
                    "train_data_len is required "
                    "when reference mode is enabled."
                )

            if self.train_data_len <= 0:
                raise ValueError(
                    "train_data_len must be positive."
                )

            if (
                self.split == "train"
                and self.data_len
                != self.train_data_len
            ):
                raise ValueError(
                    "train split length does not match "
                    "train_data_len: "
                    f"{self.data_len} vs "
                    f"{self.train_data_len}."
                )

        # ------------------------------------------------------------
        # Mask
        # ------------------------------------------------------------

        regular_settings = self.meta.get(
            "regular_settings",
            {},
        )

        null_val = regular_settings.get(
            "null_val",
            np.nan,
        )

        if null_val is None:
            nan_mask = np.zeros_like(
                self.data,
                dtype=bool,
            )
        elif (
            isinstance(null_val, float)
            and np.isnan(null_val)
        ):
            nan_mask = np.isnan(
                self.data
            )
        else:
            nan_mask = (
                self.data == null_val
            )

        mask_file = join(
            data_path,
            f"{split}_mask.npy",
        )

        if exists(mask_file):
            extra_mask = np.load(
                mask_file
            ).astype(bool)

            if (
                extra_mask.shape
                != self.data.shape
            ):
                raise ValueError(
                    f"{split}_mask.npy shape "
                    f"{extra_mask.shape} does not "
                    f"match data shape "
                    f"{self.data.shape}."
                )

            self.mask = (
                nan_mask | extra_mask
            )
        else:
            self.mask = nan_mask

        # ------------------------------------------------------------
        # Original timestamp features
        # ------------------------------------------------------------

        timestamp_file = join(
            data_path,
            f"{split}_timestamps.npy",
        )

        if exists(timestamp_file):
            timestamp = np.load(
                timestamp_file
            ).astype(np.float32)

            if timestamp.ndim == 1:
                timestamp = (
                    timestamp[:, None]
                )

            if (
                timestamp.shape[0]
                != self.data_len
            ):
                raise ValueError(
                    f"{split}_timestamps.npy length "
                    f"{timestamp.shape[0]} does not "
                    f"match data length "
                    f"{self.data_len}."
                )

            self.timestamp = timestamp
        else:
            self.timestamp = np.zeros(
                (
                    self.data_len,
                    self.timestamp_dim,
                ),
                dtype=np.float32,
            )

        self.base_num_features = (
            self.timestamp.shape[1]
        )

        self.num_features = (
            self.base_num_features
            + int(
                self.append_relative_time
            )
        )

        # ------------------------------------------------------------
        # Resolve every C length
        # ------------------------------------------------------------

        if self.reference_mode == "none":
            self.reference_lengths = []
        else:
            self.reference_lengths = [
                int(
                    np.floor(
                        self.train_data_len
                        * ratio
                        + 1e-12
                    )
                )
                for ratio
                in self.reference_ratios
            ]

            for ratio, reference_len in zip(
                self.reference_ratios,
                self.reference_lengths,
            ):
                if reference_len <= 0:
                    raise ValueError(
                        "Reference ratio resolves "
                        "to an empty C: "
                        f"ratio={ratio}, "
                        f"train_data_len="
                        f"{self.train_data_len}."
                    )

        # ------------------------------------------------------------
        # Build one global sample map
        # ------------------------------------------------------------

        self.sample_index_map: List[
            Tuple[int, int]
        ] = []

        self._build_sample_index_map()

    def _build_sample_index_map(self):
        """
        Each item is:

            (reference_index, input_start)

        reference_index == -1:
            ordinary dataset without reference bank.
        """

        final_start_exclusive = (
            self.data_len
            - self.seq_len
            + 1
        )

        if self.reference_mode == "none":
            for input_start in range(
                final_start_exclusive
            ):
                self.sample_index_map.append(
                    (-1, input_start)
                )

            return

        for (
            reference_index,
            reference_len,
        ) in enumerate(
            self.reference_lengths
        ):
            if (
                self.split == "train"
                and self.reference_mode
                == "pretrain"
            ):
                # JEPA windows begin after C.
                first_start = reference_len
            else:
                # Finetuning, validation and testing
                # use the complete local split.
                first_start = 0

            if (
                first_start
                >= final_start_exclusive
            ):
                raise ValueError(
                    "Reference ratio leaves no valid "
                    "samples: "
                    f"split={self.split}, "
                    f"ratio="
                    f"{self.reference_ratios[reference_index]}, "
                    f"reference_len={reference_len}, "
                    f"data_len={self.data_len}, "
                    f"seq_len={self.seq_len}."
                )

            for input_start in range(
                first_start,
                final_start_exclusive,
            ):
                self.sample_index_map.append(
                    (
                        reference_index,
                        input_start,
                    )
                )

    def __len__(self):
        return len(
            self.sample_index_map
        )

    def cal_stat(self):
        """
        Compute train-split statistics once.

        Do not iterate over sample_index_map, because
        later windows may appear under multiple C values.
        """

        x_mean = np.nanmean(
            self.data,
            axis=0,
            keepdims=True,
        )

        x_std = np.nanstd(
            self.data,
            axis=0,
            keepdims=True,
        )

        return (
            repeat(
                torch.from_numpy(x_mean),
                "1 n -> 1 1 n 1",
            ),
            repeat(
                torch.from_numpy(x_std),
                "1 n -> 1 1 n 1",
            ),
            self.num_features,
        )

    def _append_relative_time(
        self,
        feature_array: np.ndarray,
        local_start: int,
        local_end: int,
        reference_index: int,
    ) -> np.ndarray:
        """
        Relative timestamp:

            tau_k(g) = (g + 1) / m_k

        where:
            g   is the global zero-based index;
            m_k is the length of C_k.

        The last point of C_k has tau = 1.
        """

        reference_len = (
            self.reference_lengths[
                reference_index
            ]
        )

        global_indices = (
            self.global_time_offset
            + np.arange(
                local_start,
                local_end,
                dtype=np.float32,
            )
        )

        relative_time = (
            global_indices + 1.0
        ) / float(reference_len)

        return np.concatenate(
            [
                feature_array,
                relative_time[:, None],
            ],
            axis=1,
        ).astype(np.float32)

    def __getitem__(self, idx):
        (
            reference_index,
            input_start,
        ) = self.sample_index_map[idx]

        input_end = (
            input_start
            + self.his_len
        )

        target_end = (
            input_end
            + self.pred_len
        )

        input_seq = (
            torch.from_numpy(
                self.data[
                    input_start:input_end
                ].copy()
            )
            .unsqueeze(-1)
            .float()
        )

        target_seq = (
            torch.from_numpy(
                self.data[
                    input_end:target_end
                ].copy()
            )
            .unsqueeze(-1)
            .float()
        )

        input_feature_array = (
            self.timestamp[
                input_start:input_end
            ].copy()
        )

        target_feature_array = (
            self.timestamp[
                input_end:target_end
            ].copy()
        )

        if (
            self.append_relative_time
            and reference_index >= 0
        ):
            input_feature_array = (
                self._append_relative_time(
                    input_feature_array,
                    input_start,
                    input_end,
                    reference_index,
                )
            )

            target_feature_array = (
                self._append_relative_time(
                    target_feature_array,
                    input_end,
                    target_end,
                    reference_index,
                )
            )

        input_features = (
            torch.from_numpy(
                input_feature_array
            ).float()
        )

        target_features = (
            torch.from_numpy(
                target_feature_array
            ).float()
        )

        input_mask = (
            torch.from_numpy(
                self.mask[
                    input_start:input_end
                ].copy()
            )
            .unsqueeze(-1)
            .bool()
        )

        target_mask = (
            torch.from_numpy(
                self.mask[
                    input_end:target_end
                ].copy()
            )
            .unsqueeze(-1)
            .bool()
        )

        if reference_index < 0:
            return (
                input_seq,
                target_seq,
                input_features,
                target_features,
                input_mask,
                target_mask,
            )

        return (
            input_seq,
            target_seq,
            input_features,
            target_features,
            input_mask,
            target_mask,
            torch.tensor(
                reference_index,
                dtype=torch.long,
            ),
        )

    def get_reference_prefixes(self):
        """
        Return every train prefix used to construct
        the reference spectrum bank.
        """

        if self.split != "train":
            raise RuntimeError(
                "Reference prefixes can only be "
                "read from the train split."
            )

        if self.reference_mode == "none":
            raise RuntimeError(
                "Reference mode is disabled."
            )

        reference_sequences = []
        reference_masks = []

        for reference_len in self.reference_lengths:
            reference_seq = (
                torch.from_numpy(
                    self.data[
                        :reference_len
                    ].copy()
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .float()
            )

            reference_mask = (
                torch.from_numpy(
                    self.mask[
                        :reference_len
                    ].copy()
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .bool()
            )

            reference_sequences.append(
                reference_seq
            )

            reference_masks.append(
                reference_mask
            )

        return (
            reference_sequences,
            reference_masks,
        )