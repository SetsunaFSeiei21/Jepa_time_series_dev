"""Self-supervised pretraining; the method is the `pretrain` config group.

    python -m src.experiments.pretrain pretrain=jepa model=gwnet dataset=pems08 task=short

The run directory then holds `pretrained_model.pt`, which supervised training loads with
`train.init_checkpoint=<run dir>/pretrained_model.pt`.
"""

import hydra
from omegaconf import DictConfig

from src.training import PRETRAINERS

from .main import build_experiment


@hydra.main(version_base="1.3", config_path="../../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    exp = build_experiment(cfg)
    PRETRAINERS[cfg.pretrain.name](
        model=exp.model,
        graph=exp.graph,
        loaders=exp.loaders,
        scaler=exp.scaler,
        cfg=cfg.train,
        device=exp.device,
        save_dir=exp.save_dir,
        method=cfg.pretrain,
    ).fit()


if __name__ == "__main__":
    main()
