import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from src.utils.scalar import ZScoreNormalizer


@dataclass
class BatchData:
    """
    Container for a single batch of data.

    Attributes:
        input_seq: Input sequence tensor with shape (batch_size, seq_len, num_nodes, num_features)
        target_seq: Target sequence tensor with shape (batch_size, pred_len, num_nodes, num_features)
        input_features: Additional features for input sequence (optional)
        target_features: Additional features for target sequence (optional)
        input_mask: Boolean mask for input sequence (True = masked, optional)
        target_mask: Boolean mask for target sequence (True = masked, optional)
        metadata: Additional metadata about the batch
    """

    input_seq: torch.Tensor
    target_seq: torch.Tensor
    input_features: Optional[torch.Tensor] = None
    target_features: Optional[torch.Tensor] = None
    input_mask: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None

    # 每个样本使用 reference bank 中的哪个 C。
    # shape: (B,)
    reference_index: Optional[torch.Tensor] = None

    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate batch dimensions after initialization."""
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate that all tensors have compatible dimensions."""
        batch_size = self.input_seq.shape[0]
        num_nodes = self.input_seq.shape[2]

        # Check input_seq and target_seq batch and node dimensions
        assert (
            self.input_seq.shape[0] == self.target_seq.shape[0]
        ), f"Batch size mismatch: input_seq {self.input_seq.shape[0]} vs target_seq {self.target_seq.shape[0]}"

        assert (
            self.input_seq.shape[2] == self.target_seq.shape[2]
        ), f"Node count mismatch: input_seq {self.input_seq.shape[2]} vs target_seq {self.target_seq.shape[2]}"

        # Check optional tensors if provided
        if self.input_features is not None:
            assert (
                self.input_features.shape[0] == batch_size
            ), f"input_features batch mismatch: {self.input_features.shape[0]} vs {batch_size}"
            # assert (
            #     self.input_features.shape[2] == num_nodes
            # ), f"input_features node mismatch: {self.input_features.shape[2]} vs {num_nodes}"

        if self.target_features is not None:
            assert (
                self.target_features.shape[0] == batch_size
            ), f"target_features batch mismatch: {self.target_features.shape[0]} vs {batch_size}"
            # assert (
            #     self.target_features.shape[2] == num_nodes
            # ), f"target_features node mismatch: {self.target_features.shape[2]} vs {num_nodes}"

        if self.input_mask is not None:
            assert (
                self.input_mask.shape[:3] == self.input_seq.shape[:3]
            ), f"input_mask shape mismatch: {self.input_mask.shape} vs {self.input_seq.shape}"

        if self.target_mask is not None:
            assert (
                self.target_mask.shape[:3] == self.target_seq.shape[:3]
            ), f"target_mask shape mismatch: {self.target_mask.shape} vs {self.target_seq.shape}"
            
        if self.reference_index is not None:
            assert self.reference_index.dim() == 1, (
                "reference_index must have shape (B,), "
                f"but got {tuple(self.reference_index.shape)}."
            )

            assert self.reference_index.shape[0] == batch_size, (
                "reference_index batch size mismatch: "
                f"{self.reference_index.shape[0]} vs {batch_size}."
            )

        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.pred_len = self.target_seq.shape[1]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model input."""
        result = {
            "input_seq": self.input_seq,
            "target_seq": self.target_seq,
            "pred_len": self.pred_len,
        }

        if self.input_features is not None:
            result["input_features"] = self.input_features
        if self.target_features is not None:
            result["target_features"] = self.target_features
        if self.input_mask is not None:
            result["input_mask"] = self.input_mask
        if self.target_mask is not None:
            result["target_mask"] = self.target_mask

        if self.reference_index is not None:
            result["reference_index"] = self.reference_index

        return result

    def to_device(self, device: torch.device) -> "BatchData":
        """Move all tensors to specified device."""
        return BatchData(
            input_seq=self.input_seq.to(device),
            target_seq=self.target_seq.to(device),
            input_features=(
                self.input_features.to(device)
                if self.input_features is not None
                else None
            ),
            target_features=(
                self.target_features.to(device)
                if self.target_features is not None
                else None
            ),
            input_mask=(
                self.input_mask.to(device) if self.input_mask is not None else None
            ),
            target_mask=(
                self.target_mask.to(device)
                if self.target_mask is not None
                else None
            ),

            reference_index=(
                self.reference_index.to(device)
                if self.reference_index is not None
                else None
            ),

            metadata=self.metadata,
        )


def prepare_batch(batch_data: Union[Tuple, Dict, BatchData]) -> BatchData:
    """
    Convert various batch formats to BatchData.

    Args:
        batch_data: Can be tuple, dict, or BatchData

    Returns:
        BatchData object
    """
    if isinstance(batch_data, BatchData):
        return batch_data
    elif isinstance(batch_data, dict):
        return BatchData(**batch_data)
    elif isinstance(batch_data, tuple):
        # Handle different tuple lengths
        if len(batch_data) == 2:
            # (input_seq, target_seq)
            return BatchData(input_seq=batch_data[0], target_seq=batch_data[1])
        elif len(batch_data) == 4:
            # (input_seq, target_seq, input_features, target_features)
            return BatchData(
                input_seq=batch_data[0],
                target_seq=batch_data[1],
                input_features=batch_data[2],
                target_features=batch_data[3],
            )
        elif len(batch_data) == 6:
            # (input_seq, target_seq, input_features, target_features, input_mask, target_mask)
            return BatchData(
                input_seq=batch_data[0],
                target_seq=batch_data[1],
                input_features=batch_data[2],
                target_features=batch_data[3],
                input_mask=batch_data[4],
                target_mask=batch_data[5],
            )
        elif len(batch_data) == 7:
            return BatchData(
                input_seq=batch_data[0],
                target_seq=batch_data[1],
                input_features=batch_data[2],
                target_features=batch_data[3],
                input_mask=batch_data[4],
                target_mask=batch_data[5],
                reference_index=batch_data[6],
            )
        else:
            raise ValueError(f"Unsupported tuple length: {len(batch_data)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch_data)}")


@dataclass
class InferenceResult:
    """Container for inference results."""

    predictions: np.ndarray
    ground_truth: np.ndarray
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def save(self, save_dir: str, prefix: str = "") -> None:
        """Save results to disk."""
        os.makedirs(save_dir, exist_ok=True)

        prefix = f"{prefix}_" if prefix else ""

        np.save(os.path.join(save_dir, f"{prefix}predictions.npy"), self.predictions)
        np.save(os.path.join(save_dir, f"{prefix}ground_truth.npy"), self.ground_truth)

        # Save metrics as JSON
        if self.metrics is not None:
            with open(os.path.join(save_dir, f"{prefix}metrics.json"), "w") as f:
                json.dump({k: float(v) for k, v in self.metrics.items()}, f, indent=2)

        # Save metadata
        if self.metadata is not None:
            with open(os.path.join(save_dir, f"{prefix}metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=2)
