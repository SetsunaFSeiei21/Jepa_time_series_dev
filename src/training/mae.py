"""Masked-reconstruction (MAE) pretraining of a forecasting model's encoder.

Every node's input window is split into equal time segments and a fixed fraction of them is
hidden: the hidden entries are treated as missing, i.e. set to 0 in the model space and left out
of the per-window normalization statistics. The model encodes the masked window, a decoder maps
the latent back to a window of the same length, and the loss compares the reconstruction with
the clean window on the hidden, observed entries only.

The reconstruction decoder is the model's own `decode`, so the method works for any backbone.
By default it is a separate copy of the model that is not saved: the checkpoint then holds a
pretrained encoder and a decoder and predictor at their initialization, as after JEPA.
"""

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from einops import repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

from src.data import Batch, StandardScaler, WindowLoader
from src.graph import GraphSource
from src.models import BaseModel

from .pretrainer import Pretrainer
from .trainer import ModelInputs, model_inputs


@dataclass
class MaskedWindow(ModelInputs):
    """Model inputs whose `x` is the masked window, plus what the reconstruction is scored on."""

    target: Tensor  # (b, his_len, n, c) the clean input window in the model space
    scored: Tensor  # (b, his_len, n, c) bool, True where hidden from the model and observed


def segment_mask(
    shape: tuple[int, int, int],
    num_segments: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(b, t, n) -> bool (b, t, n, 1), True where hidden.

    The t steps of every (sample, node) are split into `num_segments` equal segments, and
    round(mask_ratio * num_segments) of them, drawn uniformly, are hidden on all channels.
    """
    b, t, n = shape
    num_hidden = round(mask_ratio * num_segments)
    noise = torch.rand(b, n, num_segments, device=device, generator=generator)
    hidden = torch.zeros_like(noise, dtype=torch.bool)
    hidden.scatter_(-1, noise.topk(num_hidden, dim=-1).indices, True)
    return repeat(hidden, "b n s -> b (s r) n 1", r=t // num_segments)


class MAETrainer(Pretrainer):
    def __init__(
        self,
        model: BaseModel,
        graph: GraphSource | None,
        loaders: dict[str, WindowLoader],
        scaler: StandardScaler,
        cfg: DictConfig,
        device: torch.device,
        save_dir: Path,
        method: DictConfig,
    ):
        assert model.his_len == model.pred_len, "the decoder reconstructs the input window"
        assert model.his_len % method.num_segments == 0, (
            f"his_len={model.his_len} is not divisible into {method.num_segments} segments"
        )
        assert 0 < round(method.mask_ratio * method.num_segments) < method.num_segments, (
            "mask_ratio must hide at least one segment and keep at least one"
        )
        self.method = method
        # Decoder used for reconstruction: a copy of the model (only its `decode` receives
        # gradients) or the model itself, whose decoder is then pretrained and saved too.
        self.reconstructor = copy.deepcopy(model).to(device) if method.separate_decoder else model
        super().__init__(model, graph, loaders, scaler, cfg, device, save_dir)
        self.criterion = instantiate(method.loss)
        self.eval_generator = torch.Generator(device=self.device)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        parameters = list(self.model.parameters())
        if self.reconstructor is not self.model:
            # Only the copy's decoder gets gradients; the optimizer skips the other parameters.
            parameters += list(self.reconstructor.parameters())
        return parameters

    def train_mode(self) -> None:
        self.model.train()
        self.reconstructor.train()

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        # The same masks at every evaluation, so validation losses are comparable across epochs.
        self.eval_generator.manual_seed(self.method.eval_seed)
        self.reconstructor.eval()
        return super().evaluate(split)

    def inputs(self, batch: Batch) -> MaskedWindow:
        generator = None if self.model.training else self.eval_generator
        b, t, n, _ = batch.x.shape
        hidden = segment_mask(
            (b, t, n), self.method.num_segments, self.method.mask_ratio, self.device, generator
        )
        # Hidden entries become missing ones (raw value 0, flagged in the mask) before anything
        # reads the batch: zero model input, and no part in the instance-normalization statistics
        # (which would otherwise reveal their mean) or in a graph computed from the batch.
        inputs = model_inputs(
            replace(batch, x=batch.x.masked_fill(hidden, 0.0), x_mask=batch.x_mask | hidden),
            self.scaler,
            self.cfg.instance_norm,
            self.graph,
        )
        target = self.scaler.transform(batch.x)
        if inputs.instance is not None:
            target = inputs.instance.transform(target)
        return MaskedWindow(
            **vars(inputs),
            target=target.masked_fill(batch.x_mask, 0.0),
            scored=hidden & ~batch.x_mask,
        )

    def loss(self, inputs: MaskedWindow) -> tuple[Tensor, dict[str, float]]:
        z = self.model.encode(inputs.x, inputs.x_mark, inputs.supports)
        # The window to reconstruct is the input window itself, hence x_mark as its time features.
        reconstruction = self.reconstructor.decode(
            z, inputs.x, inputs.x_mark, inputs.x_mark, inputs.supports
        )  # (b, his_len, n, c)
        if not inputs.scored.any():  # every hidden entry is missing: nothing to score
            return reconstruction.sum() * 0.0, {}
        return self.criterion(reconstruction[inputs.scored], inputs.target[inputs.scored]), {}
