from typing import Any, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def canonicalize_hidden_for_mae(
    hidden: Union[torch.Tensor, Sequence[torch.Tensor]],
    model_name: str,
    node_num: int,
    hidden_dim: int,
) -> torch.Tensor:
    """
    Convert different backbone encoder outputs into a canonical MAE format.

    Canonical formats:
        1. Node-aware hidden:
            (B, Tz, N, D)

        2. Global hidden without explicit node dimension:
            (B, Tz, D)

    Supported target models:
        - PatchTST:
            original: (B, N, D, patch_num)
            output:   (B, patch_num, N, D)

        - Crossformer:
            original: list of (B, N, seg_num_i, D)
            output:   (B, sum_i seg_num_i, N, D)

        - FEDformer:
            original: (B, T, D)
            output:   (B, T, D)
    """
    # Crossformer: multi-scale list
    if isinstance(hidden, (list, tuple)):
        chunks = []
        for idx, h in enumerate(hidden):
            if h.dim() != 4:
                raise ValueError(
                    f"Expected Crossformer hidden scale {idx} to be 4D "
                    f"(B,N,S,D), got {tuple(h.shape)}."
                )

            if h.size(1) != node_num:
                raise ValueError(
                    f"Expected Crossformer hidden scale {idx} node dim={node_num}, "
                    f"got shape {tuple(h.shape)}."
                )

            if h.size(-1) != hidden_dim:
                raise ValueError(
                    f"Expected Crossformer hidden scale {idx} hidden dim={hidden_dim}, "
                    f"got shape {tuple(h.shape)}."
                )

            # (B, N, S, D) -> (B, S, N, D)
            chunks.append(h.permute(0, 2, 1, 3).contiguous())

        return torch.cat(chunks, dim=1)

    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Unsupported hidden type: {type(hidden)}")

    # FEDformer: (B, T, D)
    if hidden.dim() == 3:
        if hidden.size(-1) != hidden_dim:
            raise ValueError(
                f"Expected 3D hidden last dim={hidden_dim}, got {tuple(hidden.shape)}."
            )
        return hidden

    if hidden.dim() != 4:
        raise ValueError(
            f"Expected hidden to be 3D or 4D after encode, got {tuple(hidden.shape)}."
        )

    # Already canonical: (B, T, N, D)
    if hidden.size(2) == node_num and hidden.size(-1) == hidden_dim:
        return hidden

    # PatchTST: (B, N, D, patch_num) -> (B, patch_num, N, D)
    if hidden.size(1) == node_num and hidden.size(2) == hidden_dim:
        return hidden.permute(0, 3, 1, 2).contiguous()

    # Sometimes a model may return (B, N, S, D)
    if hidden.size(1) == node_num and hidden.size(-1) == hidden_dim:
        return hidden.permute(0, 2, 1, 3).contiguous()

    raise ValueError(
        f"Cannot canonicalize hidden shape {tuple(hidden.shape)} with "
        f"node_num={node_num}, hidden_dim={hidden_dim}, model_name={model_name}."
    )


class MAEReconstructionHead(nn.Module):
    """
    Lightweight MAE reconstruction head.

    It supports two hidden formats:

    1. Node-aware hidden:
        z:      (B, Tz, N, D)
        output: (B, T,  N, 1)

        Reconstruction:
            Linear(D -> 1)

    2. Global hidden without node dimension, e.g. FEDformer:
        z:      (B, Tz, D)
        output: (B, T,  N, 1)

        Reconstruction:
            Linear(D -> N), then unsqueeze(-1)

    If Tz != target_len, temporal interpolation is used to align the
    reconstruction length to the original input sequence length.
    """

    def __init__(self, hidden_dim: int, node_num: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_num = node_num

        # For hidden with explicit node dimension: (B,Tz,N,D) -> (B,Tz,N,1)
        self.node_value_head = nn.Linear(hidden_dim, 1)

        # For hidden without explicit node dimension: (B,Tz,D) -> (B,Tz,N)
        self.global_value_head = nn.Linear(hidden_dim, node_num)

    def _resize_time(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resize temporal dimension to target_len.

        x:
            (B, Tz, N, 1)

        return:
            (B, target_len, N, 1)
        """
        if x.size(1) == target_len:
            return x

        # (B,Tz,N,1) -> (B,N,Tz)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()

        # Linear interpolation over temporal axis.
        x = F.interpolate(
            x,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

        # (B,N,T) -> (B,T,N,1)
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        return x

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        if z.dim() == 4:
            # z: (B,Tz,N,D)
            if z.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden dim={self.hidden_dim}, got {z.size(-1)}."
                )

            x_rec = self.node_value_head(z)  # (B,Tz,N,1)
            x_rec = self._resize_time(x_rec, target_len=target_len)
            return x_rec

        if z.dim() == 3:
            # z: (B,Tz,D)
            if z.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden dim={self.hidden_dim}, got {z.size(-1)}."
                )

            x_rec = self.global_value_head(z)  # (B,Tz,N)
            x_rec = x_rec.unsqueeze(-1)        # (B,Tz,N,1)
            x_rec = self._resize_time(x_rec, target_len=target_len)
            return x_rec

        raise ValueError(
            f"MAEReconstructionHead expects z to be 3D or 4D, got {tuple(z.shape)}."
        )