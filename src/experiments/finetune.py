import os

import hydra
import hydra.core
import numpy as np
import torch
from omegaconf import DictConfig

from src.engines.expert import ExpertEngine
from src.base.model import BaseModel
from src.factories.engine import TrainingEngineFactory
from src.factories.loader import get_loaders
from src.factories.logger import LoggerFactory
from src.factories.model import ModelFactory


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device,
    logger=None,
):
    """
    Load JEPA pretraining checkpoint for finetuning.

    Supported checkpoint formats:
        1. {"model_state_dict": ...}
        2. raw state_dict
    """
    if checkpoint_path is None or checkpoint_path == "":
        if logger is not None:
            logger.info("No pretrained checkpoint is provided. Finetuning from scratch.")
        else:
            print("No pretrained checkpoint is provided. Finetuning from scratch.")
        return model

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    message = (
        f"Loaded pretrained checkpoint from: {checkpoint_path}\n"
        f"Missing keys: {missing_keys}\n"
        f"Unexpected keys: {unexpected_keys}"
    )

    if logger is not None:
        logger.info(message)
    else:
        print(message)

    return model


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="default",
)
def main(cfg: DictConfig):
    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    task_exp = cfg["task"]

    seed = cfg["seed"]
    set_seed(seed)

    HIS_LEN = task_exp["his_len"]
    PRED_LEN = task_exp["pred_len"]

    device_id = exp_kwargs["device_id"]
    DEVICE = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    meta_path = cfg["meta_path"]

    loaders, adj_mtx, norm_pipeline, N_FEAT = get_loaders(
        meta_path,
        dataset_kwargs,
        exp_kwargs,
        HIS_LEN,
        PRED_LEN,
        DEVICE,
    )

    N_NODE = adj_mtx.shape[0]

    ext_kwargs = dict(
        model_kwargs,
        **task_exp,
        **{
            "node_num": N_NODE,
            "input_dim": N_FEAT + 1,
            "output_dim": 1,
            "freq": "5min",
        },
    )

    loss_func = lambda x, y: torch.nn.functional.l1_loss(
        x,
        y,
        reduction="mean",
    )

    model_name = model_kwargs["_target_"].split(".")[-1]

    model: BaseModel = ModelFactory.create_model(
        model_name,
        ext_kwargs,
        adj_mtx,
        DEVICE,
    )

    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(
                "BAD PARAM:",
                name,
                "nan:",
                torch.isnan(p).sum().item(),
                "inf:",
                torch.isinf(p).sum().item(),
                "min:",
                torch.nan_to_num(
                    p,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).min().item(),
                "max:",
                torch.nan_to_num(
                    p,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).max().item(),
            )
            breakpoint()

    log_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = LoggerFactory.create_logger(log_folder, mode="finetune")

    pretrain_ckpt = exp_kwargs.get("pretrain_ckpt", "")
    model = load_pretrained_checkpoint(
        model=model,
        checkpoint_path=pretrain_ckpt,
        device=DEVICE,
        logger=logger,
    )

    engine: ExpertEngine = TrainingEngineFactory.create_engine(
        model_name=model_name,
        device=DEVICE,
        model=model,
        logger=logger,
        save_path=log_folder,
        scalar=norm_pipeline,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
        loss_func=loss_func,
        config=exp_kwargs,
    )

    engine.run()


if __name__ == "__main__":
    main()