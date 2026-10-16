"""Contrastive pretraining of a forecasting model's encoder (frequency-masking views + InfoNCE).

Two views of every input window are made by removing random frequency components of each
node's series. Both views are encoded by the model, the latent tokens pass through a linear
projection head, and a symmetric InfoNCE loss makes the two views of the same instance more
similar (cosine similarity) than the views of all other instances of the batch.

An instance is either one latent token (`instances: token`) or a whole window, its tokens
averaged (`instances: window`). The tokens are those of `latent_tokens`, the same ones the JEPA
loss compares, so the method works for any backbone. The projection head belongs to the trainer
and is not saved: the checkpoint loads into the plain model.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor, nn

from src.data import Batch, StandardScaler, WindowLoader
from src.graph import GraphSource
from src.models import BaseModel

from .pretrainer import Pretrainer
from .trainer import ModelInputs, model_inputs

INSTANCES = ("token", "window")


@dataclass
class TwoViews(ModelInputs):
    """Model inputs whose `x` is the first augmented view of the input window."""

    x_other: Tensor  # (b, his_len, n, c) the second augmented view


def frequency_masking(
    x: Tensor,
    missing: Tensor,
    mask_ratio: float,
    preserve_dc: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(b, t, n, c) -> (b, t, n, c): every frequency of every series is removed with
    probability `mask_ratio` (the mean too unless `preserve_dc`); missing entries stay 0."""
    spectrum = torch.fft.rfft(x, dim=1)  # (b, t // 2 + 1, n, c)
    keep = torch.rand(spectrum.shape, device=x.device, generator=generator) >= mask_ratio
    if preserve_dc:
        keep[:, 0] = True
    return torch.fft.irfft(spectrum * keep, n=x.shape[1], dim=1).masked_fill(missing, 0.0)


def info_nce(a: Tensor, b: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """Symmetric InfoNCE between (m, d) embeddings whose rows i are the two views of instance i.

    Returns the loss and the fraction of rows whose most similar row of the other view is
    their own positive.
    """
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).mT / temperature  # (m, m)
    labels = torch.arange(len(logits), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.mT, labels)) / 2
    accuracy = (logits.argmax(-1) == labels).float().mean()
    return loss, accuracy


class ContrastiveTrainer(Pretrainer):
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
        assert method.instances in INSTANCES, f"instances must be one of {INSTANCES}"
        self.method = method
        width = latent_width(model.to(device), loaders["val"], scaler, cfg.instance_norm, graph)
        self.head = nn.Linear(width, width).to(device)
        super().__init__(model, graph, loaders, scaler, cfg, device, save_dir)
        self.eval_generator = torch.Generator(device=self.device)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.model.parameters()) + list(self.head.parameters())

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        # The same views at every evaluation, so validation losses are comparable across epochs.
        self.eval_generator.manual_seed(self.method.eval_seed)
        return super().evaluate(split)

    def generator(self) -> torch.Generator | None:
        return None if self.model.training else self.eval_generator

    def inputs(self, batch: Batch) -> TwoViews:
        inputs = model_inputs(batch, self.scaler, self.cfg.instance_norm, self.graph)
        views = [
            frequency_masking(
                inputs.x,
                batch.x_mask,
                self.method.freq_mask_ratio,
                self.method.preserve_dc,
                self.generator(),
            )
            for _ in range(2)
        ]
        return TwoViews(**{**vars(inputs), "x": views[0]}, x_other=views[1])

    def embed(self, x: Tensor, inputs: TwoViews) -> Tensor:
        """View (b, his_len, n, c) -> projected tokens (b, l, d)."""
        z = self.model.encode(x, inputs.x_mark, inputs.supports)
        return self.head(self.model.latent_tokens(z))

    def loss(self, inputs: TwoViews) -> tuple[Tensor, dict[str, float]]:
        a, b = self.embed(inputs.x, inputs), self.embed(inputs.x_other, inputs)  # (b, l, d)
        if self.method.instances == "window":
            a, b = a.mean(1), b.mean(1)  # (b, d)
        else:
            a, b = rearrange(a, "b l d -> (b l) d"), rearrange(b, "b l d -> (b l) d")
            # A random subset of tokens bounds the (m, m) similarity matrix.
            keep = torch.randperm(len(a), device=a.device, generator=self.generator())
            keep = keep[: self.method.max_instances]
            a, b = a[keep], b[keep]
        loss, accuracy = info_nce(a, b, self.method.temperature)
        return loss, {"accuracy": accuracy.item()}


@torch.no_grad()
def latent_width(
    model: BaseModel,
    loader: WindowLoader,
    scaler: StandardScaler,
    instance_norm: bool,
    graph: GraphSource | None,
) -> int:
    """Width d of the model's latent tokens, read from one forward pass."""
    model.eval()
    inputs = model_inputs(next(iter(loader)), scaler, instance_norm, graph)
    return model.latent_tokens(model.encode(inputs.x, inputs.x_mark, inputs.supports)).shape[-1]
