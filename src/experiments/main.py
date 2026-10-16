"""Supervised forecasting, from scratch or from a pretrained checkpoint.

python -m src.experiments.main model=gwnet dataset=pems08 task=short
python -m src.experiments.main ... train.init_checkpoint=<run>/pretrained_model.pt
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.data import (
    SpatioTemporalData,
    StandardScaler,
    WindowLoader,
    build_loaders,
    fit_dataset_scaler,
)
from src.graph import GraphSource
from src.models import BaseModel, build_graph, build_model
from src.training import Trainer

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Experiment:
    """Everything a trainer needs, built from the config."""

    model: BaseModel
    graph: GraphSource | None
    loaders: dict[str, WindowLoader]
    scaler: StandardScaler
    device: torch.device
    save_dir: Path


def build_experiment(cfg: DictConfig) -> Experiment:
    set_seed(cfg.seed)
    device, data_device = torch.device(cfg.train.device), torch.device(cfg.train.data_device)
    # Fail instead of silently training on the CPU when the configured GPU is missing.
    if "cuda" in (device.type, data_device.type) and not torch.cuda.is_available():
        raise RuntimeError(f"train.device={device} but CUDA is unavailable; set train.device=cpu")

    dataset = SpatioTemporalData.load(Path(cfg.data_root) / cfg.dataset.name, data_device)
    log.info(f"Loaded {dataset.name}: {dict(dataset.meta)}")
    loaders = build_loaders(
        dataset,
        split=cfg.dataset.split,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
        train_stride=cfg.train.train_stride,
        eval_stride=cfg.task.eval_stride,
        batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.train.eval_batch_size,
        device=device,
    )
    train_end = dataset.split_bounds(cfg.dataset.split)[0][1]
    scaler = fit_dataset_scaler(dataset.data, dataset.mask, train_end, cfg.dataset.normalize)
    graph = build_graph(cfg.model, dataset.graph.to(device) if dataset.graph is not None else None)
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        graph,
        num_nodes=dataset.num_nodes,
        in_channels=dataset.num_channels,
        out_channels=dataset.num_channels,
        time_feature_sizes=dataset.time_feature_sizes,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
    )
    return Experiment(
        model=model,
        graph=graph,
        loaders=loaders,
        scaler=scaler.to(device),
        device=device,
        save_dir=Path(HydraConfig.get().runtime.output_dir),
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    exp = build_experiment(cfg)
    Trainer(
        model=exp.model,
        graph=exp.graph,
        loaders=exp.loaders,
        scaler=exp.scaler,
        cfg=cfg.train,
        metric_space=cfg.dataset.metric_space,
        device=exp.device,
        save_dir=exp.save_dir,
    ).fit()


if __name__ == "__main__":
    main()
