# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import lejepa


# class JEPASIGRegLoss(nn.Module):
#     """
#     JEPA pretraining loss:

#         loss = MSE(Ey_pred, Ey) + alpha * SIGReg(embeddings)

#     Ey / Ey_pred shape:
#         (B, T, N, D)

#     SIGReg expects:
#         (num_samples, num_dims)

#     So we flatten:
#         (B, T, N, D) -> (B*T*N, D)
#     """

#     def __init__(
#         self,
#         alpha: float = 1.0,
#         num_points: int = 17,
#         num_slices: int = 1024,
#         detach_target: bool = True,
#         reg_on: str = "both",
#     ):
#         super().__init__()
#         self.alpha = alpha
#         self.detach_target = detach_target
#         self.reg_on = reg_on

#         univariate_test = lejepa.univariate.EppsPulley(n_points=num_points)
#         self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
#             univariate_test=univariate_test,
#             num_slices=num_slices,
#         )

#     def _flatten_embedding(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Convert hidden representation to SIGReg input.

#         Input:
#             x: (B, T, N, D)

#         Return:
#             x_flat: (B*T*N, D)
#         """
#         if x.dim() != 4:
#             raise ValueError(f"Expected embedding shape (B, T, N, D), got {x.shape}")

#         return x.reshape(-1, x.shape[-1])

#     def forward(self, Ey_pred: torch.Tensor, Ey: torch.Tensor):
#         if Ey_pred.shape != Ey.shape:
#             raise ValueError(
#                 f"Shape mismatch: Ey_pred.shape={Ey_pred.shape}, Ey.shape={Ey.shape}"
#             )

#         target = Ey.detach() if self.detach_target else Ey

#         loss_align = F.mse_loss(Ey_pred, target)

#         if self.reg_on == "pred":
#             emb = self._flatten_embedding(Ey_pred)
#             loss_reg = self.sigreg(emb)

#         elif self.reg_on == "target":
#             emb = self._flatten_embedding(target)
#             loss_reg = self.sigreg(emb)

#         elif self.reg_on == "both":
#             emb_pred = self._flatten_embedding(Ey_pred)
#             emb_target = self._flatten_embedding(target)
#             loss_reg = 0.5 * (self.sigreg(emb_pred) + self.sigreg(emb_target))

#         else:
#             raise ValueError(f"Unsupported reg_on: {self.reg_on}")

#         loss = loss_align + self.alpha * loss_reg

#         return loss, {
#             "loss": loss.detach(),
#             "loss_align": loss_align.detach(),
#             "loss_reg": loss_reg.detach(),
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F

import lejepa


class JEPASIGRegLoss(nn.Module):
    """
    JEPA pretraining loss:

        loss = alignment_loss + alpha * sigreg_loss

    Alignment loss:
        MSE(Ey_pred, stop_gradient(Ey))

    SIGReg:
        Regularizes the embedding distribution toward an isotropic
        standard Gaussian distribution.

    Args:
        alpha:
            Weight of SIGReg.

        num_points:
            Number of numerical integration points used by Epps-Pulley.

        num_slices:
            Number of random projection directions used by sliced SIGReg.

        detach_target:
            Whether to stop gradients through the target embedding.

        reg_on:
            Which embedding receives SIGReg:
                "pred"   : predicted embedding only
                "target" : target embedding only
                "both"   : average of predicted and target SIGReg

        feature_dim:
            Dimension containing the embedding features.

            Examples:
                Standard layout (B, T, N, D):
                    feature_dim = -1

                PatchTST layout (B, N, D, patch_num):
                    feature_dim = 2
    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_points: int = 17,
        num_slices: int = 1024,
        detach_target: bool = True,
        reg_on: str = "pred",
        feature_dim: int = -1,
    ):
        super().__init__()

        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        if reg_on not in {"pred", "target", "both"}:
            raise ValueError(
                f"Unsupported reg_on={reg_on}. "
                "Expected one of {'pred', 'target', 'both'}."
            )

        self.alpha = float(alpha)
        self.detach_target = bool(detach_target)
        self.reg_on = reg_on
        self.feature_dim = int(feature_dim)

        univariate_test = lejepa.univariate.EppsPulley(
            n_points=num_points,
        )

        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
        )

    def _flatten_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Move the embedding dimension to the last axis and flatten all
        remaining axes into the sample dimension.

        Example for PatchTST:

            input:
                (B, N, D, P)

            move feature_dim=2 to the last dimension:
                (B, N, P, D)

            flatten:
                (B*N*P, D)
        """
        if x.dim() < 2:
            raise ValueError(
                f"Expected embedding with at least 2 dimensions, got {x.shape}"
            )

        feature_dim = self.feature_dim

        if feature_dim < 0:
            feature_dim = x.dim() + feature_dim

        if feature_dim < 0 or feature_dim >= x.dim():
            raise ValueError(
                f"feature_dim={self.feature_dim} is invalid for "
                f"embedding shape {tuple(x.shape)}"
            )

        # Put the embedding feature dimension at the last position.
        x = torch.movedim(x, feature_dim, -1)

        embedding_dim = x.shape[-1]

        return x.reshape(-1, embedding_dim)

    def forward(
        self,
        Ey_pred: torch.Tensor,
        Ey: torch.Tensor,
    ):
        if Ey_pred.shape != Ey.shape:
            raise ValueError(
                "JEPA embedding shape mismatch: "
                f"Ey_pred.shape={tuple(Ey_pred.shape)}, "
                f"Ey.shape={tuple(Ey.shape)}"
            )

        target = Ey.detach() if self.detach_target else Ey

        # Representation prediction loss.
        loss_align = F.mse_loss(
            Ey_pred,
            target,
            reduction="mean",
        )

        if self.reg_on == "pred":
            emb_pred = self._flatten_embedding(Ey_pred)
            loss_reg = self.sigreg(emb_pred)

        elif self.reg_on == "target":
            emb_target = self._flatten_embedding(target)
            loss_reg = self.sigreg(emb_target)

        elif self.reg_on == "both":
            emb_pred = self._flatten_embedding(Ey_pred)
            emb_target = self._flatten_embedding(target)

            loss_reg = 0.5 * (
                self.sigreg(emb_pred)
                + self.sigreg(emb_target)
            )

        else:
            # This should already be prevented in __init__.
            raise RuntimeError(f"Unexpected reg_on={self.reg_on}")

        loss = loss_align + self.alpha * loss_reg

        return loss, {
            "loss": loss.detach(),
            "loss_align": loss_align.detach(),
            "loss_reg": loss_reg.detach(),
        }