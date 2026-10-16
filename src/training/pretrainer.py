import logging
import math
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from src.data import StandardScaler, WindowLoader
from src.graph import GraphSource
from src.models import BaseModel

from .trainer import (
    BYTES_PER_MB,
    ModelInputs,
    append_history,
    is_update_step,
    model_inputs,
    optimizer_step,
)

log = logging.getLogger(__name__)

FINAL_CHECKPOINT = "pretrained_model.pt"


class Pretrainer:
    """Self-supervised pretraining loop shared by all methods (JEPA, MAE, contrastive).

    A method implements `loss`, and optionally `train_mode` and `after_optimizer_step`. The loop
    runs `cfg.max_epochs` epochs with gradient accumulation, logs the training and validation
    losses, saves the model every `cfg.save_every` epochs and at the end as
    `pretrained_model.pt`, the checkpoint that supervised training loads with
    `train.init_checkpoint`. The whole model is saved; methods that add modules of their own
    (e.g. a projection head) keep them outside the model so the checkpoint loads into it.
    """

    def __init__(
        self,
        model: BaseModel,
        graph: GraphSource | None,
        loaders: dict[str, WindowLoader],
        scaler: StandardScaler,
        cfg: DictConfig,
        device: torch.device,
        save_dir: Path,
    ):
        self.model = model.to(device)
        self.graph = graph
        self.loaders = loaders
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.save_dir = save_dir
        self.history_path = save_dir / "history.jsonl"
        self.optimizer = instantiate(cfg.optimizer)(self.trainable_parameters())
        self.scheduler = instantiate(cfg.scheduler)(self.optimizer)
        self.updates_per_epoch = math.ceil(len(loaders["train"]) / cfg.accumulation_steps)
        num_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Model parameters: {num_params:,}")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters updated by the optimizer; methods with extra modules add theirs."""
        return list(self.model.parameters())

    def train_mode(self) -> None:
        self.model.train()

    def loss(self, inputs: ModelInputs) -> tuple[Tensor, dict[str, float]]:
        """Scalar loss to minimize and the named terms to log."""
        raise NotImplementedError

    def after_optimizer_step(self) -> None:
        """Hook after every parameter update (e.g. an EMA update)."""

    def inputs(self, batch) -> ModelInputs:
        return model_inputs(batch, self.scaler, self.cfg.instance_norm, self.graph)

    def train_epoch(self) -> dict[str, float]:
        """Mean loss terms of one epoch; the total is NaN if the loss diverged."""
        self.train_mode()
        loader = self.loaders["train"]
        totals: dict[str, float] = {}
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(loader, desc="pretrain", leave=False, disable=None)):
            loss, terms = self.loss(self.inputs(batch))
            if not torch.isfinite(loss):
                return {"loss": math.nan}
            (loss / self.cfg.accumulation_steps).backward()
            if is_update_step(i, len(loader), self.cfg.accumulation_steps):
                optimizer_step(self.trainable_parameters(), self.optimizer, self.cfg.clip_grad_norm)
                self.after_optimizer_step()
            for name, value in {"loss": loss.item(), **terms}.items():
                totals[name] = totals.get(name, 0.0) + value
        return {name: value / len(loader) for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        loader = self.loaders[split]
        for batch in tqdm(loader, desc=split, leave=False, disable=None):
            loss, terms = self.loss(self.inputs(batch))
            for name, value in {"loss": loss.item(), **terms}.items():
                totals[name] = totals.get(name, 0.0) + value
        return {name: value / len(loader) for name, value in totals.items()}

    def fit(self) -> Path:
        for epoch in range(1, self.cfg.max_epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train_terms = self.train_epoch()
            if math.isnan(train_terms["loss"]):
                raise RuntimeError(f"Epoch {epoch}: pretraining loss diverged.")
            val_terms = self.evaluate("val")
            self.scheduler.step()
            record = {
                "epoch": epoch,
                "lr": lr,
                "train": train_terms,
                "val": val_terms,
                "seconds": time.perf_counter() - start,
            }
            if self.device.type == "cuda":
                record["peak_memory_mb"] = (
                    torch.cuda.max_memory_allocated(self.device) / BYTES_PER_MB
                )
            append_history(self.history_path, record)
            if epoch % self.cfg.save_every == 0:
                torch.save(self.model.state_dict(), self.save_dir / f"pretrained_epoch_{epoch}.pt")
        final_path = self.save_dir / FINAL_CHECKPOINT
        torch.save(self.model.state_dict(), final_path)
        log.info(f"Saved the pretrained model to {final_path}")
        return final_path
