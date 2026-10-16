import torch
from torch import Tensor


class MaskedMetrics:
    """Accumulates errors over valid entries so that metrics are exact over a whole split."""

    def __init__(self, device: torch.device):
        self.abs_error = torch.zeros((), dtype=torch.float64, device=device)
        self.sq_error = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0

    def update(self, pred: Tensor, target: Tensor, valid: Tensor) -> None:
        error = (pred - target)[valid].double()
        self.abs_error += error.abs().sum()
        self.sq_error += error.square().sum()
        self.count += error.numel()

    def compute(self) -> dict[str, float]:
        mae = self.abs_error.item() / self.count
        mse = self.sq_error.item() / self.count
        return {"MAE": mae, "MSE": mse, "RMSE": mse**0.5}
