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


def load_adjacency_matrix(dataset_path: Path) -> np.ndarray:
    """Load adjacency matrix from dataset directory"""
    adj_path = dataset_path / "adj_mtx.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")
    return np.load(adj_path)


def collate_fn(batch):
    """
    Convert a batch of tuples to a BatchData object.

    Args:
        batch: List of tuples, each from dataset.__getitem__
            Each tuple contains:
            - input_seq: (his_len, num_nodes, 1)
            - target_seq: (pred_len, num_nodes, 1)
            - input_features: (his_len, num_features)
            - target_features: (pred_len, num_features)
            - input_mask: (his_len, num_nodes, 1)
            - target_mask: (pred_len, num_nodes, 1)

    Returns:
        BatchData object with batch dimension added
    """
    # Unpack the batch
    (
        input_seqs,
        target_seqs,
        input_feats,
        target_feats,
        input_masks,
        target_masks,
    ) = zip(*batch)

    # Stack all tensors to add batch dimension
    # input_seq: (batch_size, his_len, num_nodes, 1)
    input_seq_batch = torch.stack(input_seqs, dim=0)

    # target_seq: (batch_size, pred_len, num_nodes, 1)
    target_seq_batch = torch.stack(target_seqs, dim=0)

    # input_features: (batch_size, his_len, num_features)
    input_features_batch = torch.stack(input_feats, dim=0)

    # target_features: (batch_size, pred_len, num_features)
    target_features_batch = torch.stack(target_feats, dim=0)

    # input_mask: (batch_size, his_len, num_nodes, 1)
    input_mask_batch = torch.stack(input_masks, dim=0)

    # target_mask: (batch_size, pred_len, num_nodes, 1)
    target_mask_batch = torch.stack(target_masks, dim=0)

    # Create BatchData object
    return BatchData(
        input_seq=input_seq_batch,
        target_seq=target_seq_batch,
        input_features=input_features_batch,
        target_features=target_features_batch,
        input_mask=input_mask_batch,
        target_mask=target_mask_batch,
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
        "pin_memory": True,
        "pin_memory_device": str(device),
        "persistent_workers": True,
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


def get_loaders(
    meta_path: str,
    dataset_kwargs: Dict,
    exp_kwargs: Dict,
    his_len: int,
    pred_len: int,
    device: torch.device,
) -> Tuple[Dict[str, DataLoader], np.ndarray, int]:
    """
    Main function to get data loaders and related components

    Args:
        meta_path: Base path for datasets
        dataset_kwargs: Dataset configuration parameters
        exp_kwargs: Experiment configuration parameters
        his_len: Historical sequence length
        pred_len: Prediction sequence length
        device: Target device

    Returns:
        Tuple of (loaders, scaler, adjacency_matrix, num_features)
    """
    # 1. Load adjacency matrix
    dataset_path = Path(
        os.path.expanduser(os.path.join(meta_path, dataset_kwargs["name"]))
    )
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

    dataset_normalizer = ZScoreNormalizer(x_mean=x_mean, x_std=x_std)
    batch_normalizer = ZScoreNormalizer()
    norm_pipeline = NormalizationPipeline(
        dataset_normalizer=dataset_normalizer, batch_normalizer=batch_normalizer
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
