import os

import hydra
import hydra.core
import numpy as np
import torch
import torch.multiprocessing as mp
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


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="default",
)
def main(cfg: DictConfig):
    # mp.set_start_method("spawn", force=True)
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
        meta_path, dataset_kwargs, exp_kwargs, HIS_LEN, PRED_LEN, DEVICE
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
    loss_func = lambda x, y: torch.nn.functional.l1_loss(x, y, reduction="mean")
    model_name = model_kwargs["_target_"].split(".")[-1]
    model: BaseModel = ModelFactory.create_model(
        model_name, ext_kwargs, adj_mtx, DEVICE
    )
    # model.load_state_dict(torch.load("outputs/2026-01-24/11-01-15/model_parameters.pt"))
    log_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = LoggerFactory.create_logger(log_folder, mode="train")
    # engine: ExpertEngine = TrainingEngineFactory.create_engine(
    #     model_name=model_name,
    #     device=DEVICE,
    #     model=model,
    #     logger=logger,
    #     save_path=log_folder,
    #     scalar=norm_pipeline,
    #     train_loader=loaders["train_loader"],
    #     val_loader=loaders["val_loader"],
    #     test_loader=loaders["test_loader"],
    #     loss_func=loss_func,
    #     config=exp_kwargs,
    # )

    # engine.run()
    train_mode = exp_kwargs.get("train_mode", "finetune")

    if train_mode == "pretrain":
        from src.engines.jepa_pretrain import JEPAPretrainEngine

        engine = JEPAPretrainEngine(
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

    elif train_mode == "finetune":
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

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    engine.run()


if __name__ == "__main__":
    main()
