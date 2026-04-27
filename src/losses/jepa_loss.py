import torch
import torch.nn as nn
import torch.nn.functional as F

import lejepa


class JEPASIGRegLoss(nn.Module):
    """
    JEPA pretraining loss:

        loss = MSE(Ey_pred, Ey) + alpha * SIGReg(embeddings)

    Ey / Ey_pred shape:
        (B, T, N, D)

    SIGReg expects:
        (num_samples, num_dims)

    So we flatten:
        (B, T, N, D) -> (B*T*N, D)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_points: int = 17,
        num_slices: int = 1024,
        detach_target: bool = True,
        reg_on: str = "both",
    ):
        super().__init__()
        self.alpha = alpha
        self.detach_target = detach_target
        self.reg_on = reg_on

        univariate_test = lejepa.univariate.EppsPulley(n_points=num_points)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
        )

    def _flatten_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden representation to SIGReg input.

        Input:
            x: (B, T, N, D)

        Return:
            x_flat: (B*T*N, D)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected embedding shape (B, T, N, D), got {x.shape}")

        return x.reshape(-1, x.shape[-1])

    def forward(self, Ey_pred: torch.Tensor, Ey: torch.Tensor):
        if Ey_pred.shape != Ey.shape:
            raise ValueError(
                f"Shape mismatch: Ey_pred.shape={Ey_pred.shape}, Ey.shape={Ey.shape}"
            )

        target = Ey.detach() if self.detach_target else Ey

        loss_align = F.mse_loss(Ey_pred, target)

        if self.reg_on == "pred":
            emb = self._flatten_embedding(Ey_pred)
            loss_reg = self.sigreg(emb)

        elif self.reg_on == "target":
            emb = self._flatten_embedding(target)
            loss_reg = self.sigreg(emb)

        elif self.reg_on == "both":
            emb_pred = self._flatten_embedding(Ey_pred)
            emb_target = self._flatten_embedding(target)
            loss_reg = 0.5 * (self.sigreg(emb_pred) + self.sigreg(emb_target))

        else:
            raise ValueError(f"Unsupported reg_on: {self.reg_on}")

        loss = loss_align + self.alpha * loss_reg

        return loss, {
            "loss": loss.detach(),
            "loss_align": loss_align.detach(),
            "loss_reg": loss_reg.detach(),
        }