import torch
import torch.nn.functional as F


def masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    Compute MAE-style reconstruction loss only on artificially masked positions.

    Args:
        pred:
            Reconstructed sequence, shape (B,T,N,1).

        target:
            Original clean sequence in the same normalized space, shape (B,T,N,1).

        mask:
            Boolean mask, shape (B,T,N,1).
            True means this position is artificially masked and should contribute
            to the MAE reconstruction loss.

        loss_type:
            "l1" or "mse".

    Return:
        scalar loss.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred.shape={pred.shape}, target.shape={target.shape}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            f"Mask shape mismatch: mask.shape={mask.shape}, target.shape={target.shape}"
        )

    if mask.dtype != torch.bool:
        mask = mask.bool()

    valid_count = mask.sum()

    if valid_count.item() == 0:
        raise ValueError("masked_mae_loss receives an empty mask.")

    pred_masked = pred[mask]
    target_masked = target[mask]

    if loss_type == "l1":
        return F.l1_loss(pred_masked, target_masked, reduction="mean")

    if loss_type == "mse":
        return F.mse_loss(pred_masked, target_masked, reduction="mean")

    raise ValueError(f"Unsupported MAE loss_type: {loss_type}")