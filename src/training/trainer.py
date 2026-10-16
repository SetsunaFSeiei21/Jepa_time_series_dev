import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm import tqdm

from src.data import Batch, StandardScaler, WindowLoader
from src.graph import GraphSource
from src.models import BaseModel

from .metrics import MaskedMetrics

log = logging.getLogger(__name__)

BYTES_PER_MB = 1024**2
# Floor of the per-window standard deviation, in units of the dataset z-score (std 1): a nearly
# flat input window is only centered, not scaled up, so the target window normalized with its
# statistics stays bounded (otherwise values reach ~1e3 on PEMS08 and METR-LA training windows).
INSTANCE_MIN_STD = 1.0
# Progress bars use disable=None: shown on an interactive terminal, silent in logs and pipes.


@dataclass
class ModelInputs:
    """A batch in the model space: normalized, missing entries set to 0."""

    x: Tensor  # (b, his_len, n, c)
    y: Tensor  # (b, pred_len, n, c), normalized like x (used as a target window by pretraining)
    x_mark: Tensor  # (b, his_len, f)
    y_mark: Tensor  # (b, pred_len, f)
    supports: Tensor | None  # (b, k, n, n)
    instance: StandardScaler | None  # per-window statistics of x, to map predictions back


def model_inputs(
    batch: Batch, scaler: StandardScaler, instance_norm: bool, graph: GraphSource | None
) -> ModelInputs:
    """Dataset z-score, then (optionally) a z-score of every input window over time.

    The target window is normalized with the statistics of its input window, so both windows
    live in the same space and nothing about the target leaks into the input. The window
    standard deviation is floored at `INSTANCE_MIN_STD`.
    """
    x, y = scaler.transform(batch.x), scaler.transform(batch.y)
    instance = None
    if instance_norm:
        instance = StandardScaler.fit(x, batch.x_mask, dims=(1,))
        instance = StandardScaler(instance.mean, instance.std.clamp(min=INSTANCE_MIN_STD))
        x, y = instance.transform(x), instance.transform(y)
    return ModelInputs(
        x=x.masked_fill(batch.x_mask, 0.0),
        y=y.masked_fill(batch.y_mask, 0.0),
        x_mark=batch.x_mark,
        y_mark=batch.y_mark,
        supports=graph(batch) if graph is not None else None,
        instance=instance,
    )


def optimizer_step(
    parameters: list[nn.Parameter], optimizer: torch.optim.Optimizer, clip_grad_norm: float
) -> None:
    """Clips the gradient norm of `parameters` (0 disables clipping), steps and resets."""
    if clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()


def is_update_step(index: int, num_batches: int, accumulation_steps: int) -> bool:
    """Whether micro-batch `index` (0-based) closes a gradient-accumulation group."""
    return (index + 1) % accumulation_steps == 0 or index + 1 == num_batches


def append_history(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(json.dumps(record))


def load_weights(model: nn.Module, path: str | Path, device: torch.device) -> None:
    """Loads a state dict saved by a trainer of this repository (same architecture)."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    log.info(f"Initialized the model from {path}")


class Trainer:
    """Supervised forecasting: trains on the train split, selects the best epoch on val, and
    reports test metrics. The model starts from `cfg.init_checkpoint` when it is set (e.g. a
    pretrained encoder), otherwise from scratch."""

    def __init__(
        self,
        model: BaseModel,
        graph: GraphSource | None,
        loaders: dict[str, WindowLoader],
        scaler: StandardScaler,
        cfg: DictConfig,
        metric_space: str,
        device: torch.device,
        save_dir: Path,
    ):
        assert metric_space in ("raw", "normalized"), metric_space
        self.model = model.to(device)
        if cfg.init_checkpoint is not None:
            load_weights(self.model, cfg.init_checkpoint, device)
        self.graph = graph
        self.loaders = loaders
        self.scaler = scaler
        self.cfg = cfg
        self.metric_space = metric_space
        self.device = device
        self.loss_fn = instantiate(cfg.loss)
        self.optimizer = instantiate(cfg.optimizer)(self.model.parameters())
        self.scheduler = instantiate(cfg.scheduler)(self.optimizer)
        self.best_path = save_dir / "best_model.pt"
        self.history_path = save_dir / "history.jsonl"
        self.test_path = save_dir / "test_metrics.json"
        num_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Model parameters: {num_params:,}")

    def predict(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Prediction and target, both in the metric space."""
        inputs = model_inputs(batch, self.scaler, self.cfg.instance_norm, self.graph)
        pred = self.model(inputs.x, inputs.x_mark, inputs.y_mark, inputs.supports)
        if inputs.instance is not None:
            pred = inputs.instance.inverse_transform(pred)
        if self.metric_space == "normalized":
            return pred, self.scaler.transform(batch.y)
        return self.scaler.inverse_transform(pred), batch.y

    def train_epoch(self) -> float:
        """Mean training loss of one epoch, or NaN if the loss diverged."""
        self.model.train()
        loader = self.loaders["train"]
        total_loss, num_batches = 0.0, 0
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(loader, desc="train", leave=False, disable=None)):
            valid = ~batch.y_mask
            if not valid.any():
                continue
            pred, target = self.predict(batch)
            loss = self.loss_fn(pred[valid], target[valid])
            if not torch.isfinite(loss):
                return math.nan
            (loss / self.cfg.accumulation_steps).backward()
            if is_update_step(i, len(loader), self.cfg.accumulation_steps):
                parameters = list(self.model.parameters())
                optimizer_step(parameters, self.optimizer, self.cfg.clip_grad_norm)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        self.model.eval()
        metrics = MaskedMetrics(self.device)
        for batch in tqdm(self.loaders[split], desc=split, leave=False, disable=None):
            pred, target = self.predict(batch)
            metrics.update(pred, target, ~batch.y_mask)
        return metrics.compute()

    def fit(self) -> dict[str, float]:
        best_score = math.inf
        for epoch in range(1, self.cfg.max_epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = self.train_epoch()
            if math.isnan(train_loss):
                log.error(f"Epoch {epoch}: training loss diverged, stopping.")
                break
            val_metrics = self.evaluate("val")
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val": val_metrics,
                "seconds": time.perf_counter() - start,
            }
            if self.device.type == "cuda":
                record["peak_memory_mb"] = (
                    torch.cuda.max_memory_allocated(self.device) / BYTES_PER_MB
                )
            append_history(self.history_path, record)

            score = val_metrics[self.cfg.monitor]
            if score < best_score:
                best_score = score
                torch.save(self.model.state_dict(), self.best_path)
                log.info(f"Epoch {epoch}: new best val {self.cfg.monitor} {score:.4f}")

        if not self.best_path.exists():
            raise RuntimeError("Training produced no valid epoch to evaluate.")
        load_weights(self.model, self.best_path, self.device)
        test_metrics = self.evaluate("test")
        with open(self.test_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        log.info(f"Test metrics of the best model: {test_metrics}")
        return test_metrics
