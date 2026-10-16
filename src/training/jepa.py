"""JEPA pretraining of a forecasting model's encoder and latent predictor.

The input window is encoded by the trained ("online") model and the window that follows it by
a target encoder, an exponential moving average (EMA) of the online model that receives no
gradient. The online model's predictor learns to predict the target latent from the input
latent (alignment loss, mean squared error), and SIGReg keeps the predicted latents close to
an isotropic Gaussian so that they cannot collapse to a constant.
"""

import copy
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor, nn

from .pretrainer import Pretrainer
from .sigreg import SIGReg
from .trainer import ModelInputs


def disable_dropout(model: nn.Module) -> None:
    """Puts every dropout layer in eval mode and leaves the rest of the model as it is."""
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.eval()


class JEPATrainer(Pretrainer):
    def __init__(self, *args, method: DictConfig, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.model
        assert model.his_len == model.pred_len, "JEPA encodes input and target windows alike"
        self.method = method
        self.target = copy.deepcopy(model).requires_grad_(False).eval()
        self.sigreg = SIGReg(
            num_slices=method.sigreg.num_slices,
            t_max=method.sigreg.t_max,
            num_points=method.sigreg.num_points,
        ).to(self.device)
        self.total_updates = self.updates_per_epoch * self.cfg.max_epochs
        self.num_updates = 0
        # Running mean of the target tokens (l, d), see `center`.
        self.token_mean: Tensor | None = None

    def train_mode(self) -> None:
        self.model.train()
        # The target encoder runs without dropout. Dropout in the online encoder would add
        # noise to the predicted latent that no predictor can remove, i.e. a floor under the
        # alignment loss, so it is switched off here; supervised training keeps it.
        if self.method.disable_online_dropout:
            disable_dropout(self.model)

    def loss(self, inputs: ModelInputs) -> tuple[Tensor, dict[str, float]]:
        model = self.model
        predicted = model.latent_tokens(
            model.predict(model.encode(inputs.x, inputs.x_mark, inputs.supports))
        )
        with torch.no_grad():
            target = self.target.latent_tokens(
                self.target.encode(inputs.y, inputs.y_mark, inputs.supports)
            )
        if self.method.centering.enabled:
            predicted, target = self.center(predicted, target)
        alignment = F.mse_loss(predicted, target)
        regularization = self.sigreg(rearrange(predicted, "b l d -> (b l) d"))
        loss = alignment + self.method.sigreg_weight * regularization
        # Mean standard deviation of the predicted latent channels: drops towards 0 on collapse.
        spread = rearrange(predicted, "b l d -> (b l) d").std(0).mean()
        return loss, {
            "alignment": alignment.item(),
            "sigreg": regularization.item(),
            "latent_std": spread.item(),
        }

    @torch.no_grad()
    def update_token_mean(self, target: Tensor) -> Tensor:
        """Running mean over training batches of each target token position: (b, l, d) -> (l, d)"""
        batch_mean = target.mean(0)
        if self.token_mean is None:
            self.token_mean = batch_mean
        elif self.model.training:
            self.token_mean.lerp_(batch_mean, 1 - self.method.centering.momentum)
        return self.token_mean

    def center(self, predicted: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Removes the part of every token that is the same for all windows.

        A model whose tokens carry a large input-independent component (e.g. a learned
        positional embedding per token) can lower the alignment loss by reproducing that
        component instead of the content of the window. Subtracting the expected value of each
        token position, estimated by a running mean of the target tokens over training batches
        (independent of the micro-batch size), makes both the alignment and SIGReg act on what
        depends on the window only. Evaluation uses the running mean without updating it.
        """
        token_mean = self.update_token_mean(target)
        return predicted - token_mean, target - token_mean

    def ema_momentum(self) -> float:
        """Cosine schedule from `ema.start` (first update) to `ema.end` (last update)."""
        start, end = self.method.ema.start, self.method.ema.end
        progress = min(self.num_updates / max(self.total_updates - 1, 1), 1.0)
        return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        momentum = self.ema_momentum()
        for target, online in zip(self.target.parameters(), self.model.parameters(), strict=True):
            target.lerp_(online, 1 - momentum)
        for target, online in zip(self.target.buffers(), self.model.buffers(), strict=True):
            target.copy_(online)
        self.num_updates += 1
