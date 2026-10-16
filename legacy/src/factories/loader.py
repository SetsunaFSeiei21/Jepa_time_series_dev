import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Subset

from src.base.data import BatchData
from src.dataset.mts import MTSDataset
from src.dataset.presplit_mts import PreSplitMTSDataset
from src.utils.scalar import NormalizationPipeline, ZScoreNormalizer


@dataclass
class DatasetConfig:
    """Dataset configuration parameters"""

    name: str
    train_val_point: Union[int, float]
    val_test_point: Union[int, float]


@dataclass
class DataLoaderConfig:
    """DataLoader configuration parameters"""

    num_workers: int
    prefetch_factor: Optional[int] = None


# def load_adjacency_matrix(dataset_path: Path) -> np.ndarray:
#     """Load adjacency matrix from dataset directory"""
#     adj_path = dataset_path / "adj_mtx.npy"
#     if not adj_path.exists():
#         raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")
#     return np.load(adj_path)

def load_adjacency_matrix(dataset_path: Path, num_nodes: Optional[int] = None) -> np.ndarray:
    """
    Load adjacency matrix from dataset directory.

    For graph datasets:
        read adj_mtx.npy.

    For non-graph pre-split time-series datasets:
        if adj_mtx.npy does not exist, return identity matrix.
    """
    adj_path = dataset_path / "adj_mtx.npy"

    if adj_path.exists():
        return np.load(adj_path)

    if num_nodes is None:
        raise FileNotFoundError(
            f"Adjacency matrix not found at {adj_path}, and num_nodes is None. "
            f"Cannot create fallback identity adjacency."
        )

    return np.eye(num_nodes, dtype=np.float32)


def collate_fn(batch):
    """
    Supports:

    1. ordinary six-item samples;
    2. multi-reference seven-item samples.
    """

    sample_size = len(batch[0])

    if sample_size == 6:
        (
            input_seqs,
            target_seqs,
            input_feats,
            target_feats,
            input_masks,
            target_masks,
        ) = zip(*batch)

        reference_index_batch = None

    elif sample_size == 7:
        (
            input_seqs,
            target_seqs,
            input_feats,
            target_feats,
            input_masks,
            target_masks,
            reference_indices,
        ) = zip(*batch)

        reference_index_batch = (
            torch.stack(
                reference_indices,
                dim=0,
            ).long()
        )

    else:
        raise ValueError(
            "Unsupported dataset sample size: "
            f"{sample_size}."
        )

    return BatchData(
        input_seq=torch.stack(
            input_seqs,
            dim=0,
        ),
        target_seq=torch.stack(
            target_seqs,
            dim=0,
        ),
        input_features=torch.stack(
            input_feats,
            dim=0,
        ),
        target_features=torch.stack(
            target_feats,
            dim=0,
        ),
        input_mask=torch.stack(
            input_masks,
            dim=0,
        ),
        target_mask=torch.stack(
            target_masks,
            dim=0,
        ),
        reference_index=(
            reference_index_batch
        ),
    )


def cal_set_length(
    dataset_length: int,
    train_point: Union[int, float],
    train_val_point: Union[int, float],
    val_test_point: Union[int, float],
):
    # Convert ratio to absolute index if needed
    if isinstance(train_point, float):
        if not 0 <= train_point < 1:
            raise ValueError(
                f"train_val_point ratio must be in (0, 1), got {train_point}"
            )
        train_point = int(dataset_length * train_point)

    if isinstance(train_val_point, float):
        if not 0 < train_val_point < 1:
            raise ValueError(
                f"train_val_point ratio must be in (0, 1), got {train_val_point}"
            )
        train_val_point = int(dataset_length * train_val_point)

    if isinstance(val_test_point, float):
        if not 0 < val_test_point < 1:
            raise ValueError(
                f"val_test_point ratio must be in (0, 1), got {val_test_point}"
            )
        val_test_point = int(dataset_length * val_test_point)

    return train_point, train_val_point, val_test_point


def split_dataset_indices(
    dataset_length: int,
    train_point: Union[int, float],
    train_val_point: Union[int, float],
    val_test_point: Union[int, float],
    his_len: int,
    pred_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split dataset into train/validation/test indices

    Args:
        dataset_length: Total length of dataset
        train_val_point: Training set end point (int or float ratio)
        val_test_point: Validation set end point (int or float ratio)
        his_len: Historical sequence length
        pred_len: Prediction sequence length

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """

    # Generate indices for each split
    train_indices = torch.arange(
        start=train_point, end=train_val_point - his_len - pred_len, step=1
    )

    val_indices = torch.arange(
        start=train_val_point - his_len,
        end=val_test_point - his_len - pred_len,
        step=pred_len,
    )

    test_indices = torch.arange(
        start=val_test_point - his_len, end=dataset_length, step=pred_len
    )

    return train_indices, val_indices, test_indices


def create_data_loaders(
    dataset: MTSDataset,
    indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    loader_config: DataLoaderConfig,
    device: torch.device,
) -> Dict[str, DataLoader]:
    """
    Create DataLoader instances for train/validation/test splits

    Args:
        dataset: The base dataset
        indices: Tuple of (train_indices, val_indices, test_indices)
        loader_config: DataLoader configuration
        device: Target device for pin_memory

    Returns:
        Dictionary containing train_loader, val_loader, and test_loader
    """
    train_indices, val_indices, test_indices = indices

    # Create Subset instances
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # Common DataLoader configuration
    loader_kwargs = {
        "num_workers": loader_config.num_workers,
        "prefetch_factor": loader_config.prefetch_factor,
        # "pin_memory": True,
        # "pin_memory_device": str(device),
        # "persistent_workers": True,
        "pin_memory": False,
        "persistent_workers": False
    }

    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

def create_presplit_data_loaders(
    train_dataset: PreSplitMTSDataset,
    val_dataset: PreSplitMTSDataset,
    test_dataset: PreSplitMTSDataset,
    batch_size: int,
    loader_config: DataLoaderConfig,
    device: torch.device,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for pre-split datasets.

    Scheme A:
        train_loader reads train_data.npy only.
        val_loader reads val_data.npy only.
        test_loader reads test_data.npy only.
    """
    loader_kwargs = {
        "num_workers": loader_config.num_workers,
        "prefetch_factor": loader_config.prefetch_factor,
        "pin_memory": False,
        "persistent_workers": False,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def get_presplit_loaders(
    meta_path: str,
    dataset_kwargs: Dict,
    exp_kwargs: Dict,
    his_len: int,
    pred_len: int,
    device: torch.device,
) -> Tuple[Dict[str, DataLoader], np.ndarray, NormalizationPipeline, int]:
    """
    Loader for pre-split time-series datasets.

    Expected dataset format:
        dataset_path/
        ├── meta.json
        ├── train_data.npy
        ├── train_timestamps.npy
        ├── val_data.npy
        ├── val_timestamps.npy
        ├── test_data.npy
        └── test_timestamps.npy

    This function does NOT split by train_point / train_val_point / val_test_point.
    """

    # Allow dataset yaml to override the global meta_path.
    # This is useful because STFM datasets and time-series datasets live in
    # different root directories.
    data_root = dataset_kwargs.get("data_root", meta_path)

    dataset_path = Path(
        os.path.expanduser(os.path.join(data_root, dataset_kwargs["name"]))
    )

    timestamp_dim = dataset_kwargs.get(
        "timestamp_dim",
        4,
    )

    # ------------------------------------------------------------
    # Reference-bank configuration
    # ------------------------------------------------------------

    use_reference_bank = bool(
        exp_kwargs.get(
            "use_reference_bank",
            False,
        )
    )

    train_mode = str(
        exp_kwargs.get(
            "train_mode",
            "finetune",
        )
    ).lower()

    if use_reference_bank:
        if train_mode == "pretrain":
            reference_ratios = list(
                exp_kwargs.get(
                    "train_ratios",
                    [],
                )
            )

            reference_mode = "pretrain"

        elif train_mode == "finetune":
            reference_ratios = list(
                exp_kwargs.get(
                    "checkpoint_ratios",
                    [],
                )
            )

            reference_mode = "finetune"

        else:
            raise ValueError(
                "Reference-bank experiments require "
                "train_mode='pretrain' or "
                f"'finetune', got {train_mode}."
            )

        append_relative_time = bool(
            exp_kwargs.get(
                "append_relative_time",
                True,
            )
        )

    else:
        # Preserve all original experiments.
        reference_ratios = []
        reference_mode = "none"
        append_relative_time = False

    train_data_len = int(
        np.load(
            dataset_path
            / "train_data.npy",
            mmap_mode="r",
        ).shape[0]
    )

    val_data_len = int(
        np.load(
            dataset_path
            / "val_data.npy",
            mmap_mode="r",
        ).shape[0]
    )

    train_dataset = PreSplitMTSDataset(
        dataset_path,
        his_len=his_len,
        pred_len=pred_len,
        split="train",
        timestamp_dim=timestamp_dim,
        reference_ratios=(
            reference_ratios
        ),
        reference_mode=(
            reference_mode
        ),
        append_relative_time=(
            append_relative_time
        ),
        global_time_offset=0,
        train_data_len=(
            train_data_len
        ),
    )

    val_dataset = PreSplitMTSDataset(
        dataset_path,
        his_len=his_len,
        pred_len=pred_len,
        split="val",
        timestamp_dim=timestamp_dim,
        reference_ratios=(
            reference_ratios
        ),
        reference_mode=(
            reference_mode
        ),
        append_relative_time=(
            append_relative_time
        ),
        global_time_offset=(
            train_data_len
        ),
        train_data_len=(
            train_data_len
        ),
    )

    test_dataset = PreSplitMTSDataset(
        dataset_path,
        his_len=his_len,
        pred_len=pred_len,
        split="test",
        timestamp_dim=timestamp_dim,
        reference_ratios=(
            reference_ratios
        ),
        reference_mode=(
            reference_mode
        ),
        append_relative_time=(
            append_relative_time
        ),
        global_time_offset=(
            train_data_len
            + val_data_len
        ),
        train_data_len=(
            train_data_len
        ),
    )

    # Non-graph time-series datasets do not have adj_mtx.npy.
    # Return identity adjacency so the rest of the framework can still receive
    # an adjacency matrix.
    adj_mtx = load_adjacency_matrix(
        dataset_path,
        num_nodes=train_dataset.num_nodes,
    )
    np.fill_diagonal(adj_mtx, 1)

    # Normalization:
    # Use train split only. This avoids val/test leakage.
    if dataset_kwargs.get("use_dataset_norm", True):
        x_mean, x_std, num_features = train_dataset.cal_stat()
        x_std = x_std.clamp(min=1)
        x_mean, x_std = x_mean.to(device), x_std.to(device)
    else:
        x_mean = torch.zeros(1, 1, train_dataset.num_nodes, 1, device=device)
        x_std = torch.ones(1, 1, train_dataset.num_nodes, 1, device=device)
        num_features = train_dataset.num_features

    dataset_normalizer = ZScoreNormalizer(x_mean=x_mean, x_std=x_std)
    batch_normalizer = ZScoreNormalizer()
    norm_pipeline = NormalizationPipeline(
        dataset_normalizer=dataset_normalizer,
        batch_normalizer=batch_normalizer,
    )

    loader_config = DataLoaderConfig(
        num_workers=exp_kwargs["num_workers"],
        prefetch_factor=exp_kwargs["prefetch_factor"],
    )

    loaders = create_presplit_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=exp_kwargs["batch_size"],
        loader_config=loader_config,
        device=device,
    )

    return loaders, adj_mtx, norm_pipeline, num_features

# def get_loaders(
#     meta_path: str,
#     dataset_kwargs: Dict,
#     exp_kwargs: Dict,
#     his_len: int,
#     pred_len: int,
#     device: torch.device,
# ) -> Tuple[Dict[str, DataLoader], np.ndarray, int]:
#     """
#     Main function to get data loaders and related components

#     Args:
#         meta_path: Base path for datasets
#         dataset_kwargs: Dataset configuration parameters
#         exp_kwargs: Experiment configuration parameters
#         his_len: Historical sequence length
#         pred_len: Prediction sequence length
#         device: Target device

#     Returns:
#         Tuple of (loaders, scaler, adjacency_matrix, num_features)
#     """
#     # 1. Load adjacency matrix
#     dataset_path = Path(
#         os.path.expanduser(os.path.join(meta_path, dataset_kwargs["name"]))
#     )
#     adj_mtx = load_adjacency_matrix(dataset_path)
#     np.fill_diagonal(adj_mtx, 1)

#     # 2. Create dataset
#     dataset = MTSDataset(dataset_path, his_len, pred_len)
#     train_point, train_val_point, val_test_point = cal_set_length(
#         len(dataset),
#         dataset_kwargs["train_point"],
#         dataset_kwargs["train_val_point"],
#         dataset_kwargs["val_test_point"],
#     )
#     if dataset_kwargs["use_dataset_norm"]:
#         x_mean, x_std, num_features = dataset.cal_stat(
#             train_point,
#             train_val_point,
#         )
#         x_std = x_std.clamp(min=1)
#         x_mean, x_std = x_mean.to(device), x_std.to(device)

#     dataset_normalizer = ZScoreNormalizer(x_mean=x_mean, x_std=x_std)
#     batch_normalizer = ZScoreNormalizer()
#     norm_pipeline = NormalizationPipeline(
#         dataset_normalizer=dataset_normalizer, batch_normalizer=batch_normalizer
#     )

#     # 3. Split dataset
#     indices = split_dataset_indices(
#         dataset_length=len(dataset),
#         train_point=train_point,
#         train_val_point=train_val_point,
#         val_test_point=val_test_point,
#         his_len=his_len,
#         pred_len=pred_len,
#     )

#     # 4. Create data loaders
#     loader_config = DataLoaderConfig(
#         num_workers=exp_kwargs["num_workers"],
#         prefetch_factor=exp_kwargs["prefetch_factor"],
#     )

#     loaders = create_data_loaders(
#         dataset=dataset,
#         indices=indices,
#         batch_size=exp_kwargs["batch_size"],
#         loader_config=loader_config,
#         device=device,
#     )

#     return loaders, adj_mtx, norm_pipeline, num_features


def get_loaders(
    meta_path: str,
    dataset_kwargs: Dict,
    exp_kwargs: Dict,
    his_len: int,
    pred_len: int,
    device: torch.device,
) -> Tuple[Dict[str, DataLoader], np.ndarray, NormalizationPipeline, int]:
    """
    Main function to get data loaders and related components.

    Supports two dataset formats:

    1. format: full
        Old STFM format:
            dataset_path/
            ├── data.npy
            ├── dataset_config.json
            ├── adj_mtx.npy
            └── mask.npy

        This branch uses MTSDataset and splits by train_point /
        train_val_point / val_test_point.

    2. format: presplit
        New time-series format:
            dataset_path/
            ├── meta.json
            ├── train_data.npy
            ├── val_data.npy
            └── test_data.npy

        This branch uses PreSplitMTSDataset and does not re-split.
    """

    dataset_format = dataset_kwargs.get("format", "full")

    if dataset_format == "presplit":
        return get_presplit_loaders(
            meta_path=meta_path,
            dataset_kwargs=dataset_kwargs,
            exp_kwargs=exp_kwargs,
            his_len=his_len,
            pred_len=pred_len,
            device=device,
        )

    if dataset_format != "full":
        raise ValueError(
            f"Unsupported dataset format: {dataset_format}. "
            f"Expected 'full' or 'presplit'."
        )

    # ================================
    # Old STFM branch
    # ================================
    data_root = dataset_kwargs.get("data_root", meta_path)

    dataset_path = Path(
        os.path.expanduser(os.path.join(data_root, dataset_kwargs["name"]))
    )

    # 1. Load adjacency matrix
    adj_mtx = load_adjacency_matrix(dataset_path)
    np.fill_diagonal(adj_mtx, 1)

    # 2. Create dataset
    dataset = MTSDataset(dataset_path, his_len, pred_len)

    train_point, train_val_point, val_test_point = cal_set_length(
        len(dataset),
        dataset_kwargs["train_point"],
        dataset_kwargs["train_val_point"],
        dataset_kwargs["val_test_point"],
    )

    if dataset_kwargs["use_dataset_norm"]:
        x_mean, x_std, num_features = dataset.cal_stat(
            train_point,
            train_val_point,
        )
        x_std = x_std.clamp(min=1)
        x_mean, x_std = x_mean.to(device), x_std.to(device)
    else:
        x_mean = torch.zeros(1, 1, dataset.num_nodes, 1, device=device)
        x_std = torch.ones(1, 1, dataset.num_nodes, 1, device=device)
        num_features = dataset.num_features

    dataset_normalizer = ZScoreNormalizer(x_mean=x_mean, x_std=x_std)
    batch_normalizer = ZScoreNormalizer()
    norm_pipeline = NormalizationPipeline(
        dataset_normalizer=dataset_normalizer,
        batch_normalizer=batch_normalizer,
    )

    # 3. Split dataset
    indices = split_dataset_indices(
        dataset_length=len(dataset),
        train_point=train_point,
        train_val_point=train_val_point,
        val_test_point=val_test_point,
        his_len=his_len,
        pred_len=pred_len,
    )

    # 4. Create data loaders
    loader_config = DataLoaderConfig(
        num_workers=exp_kwargs["num_workers"],
        prefetch_factor=exp_kwargs["prefetch_factor"],
    )

    loaders = create_data_loaders(
        dataset=dataset,
        indices=indices,
        batch_size=exp_kwargs["batch_size"],
        loader_config=loader_config,
        device=device,
    )

    return loaders, adj_mtx, norm_pipeline, num_features