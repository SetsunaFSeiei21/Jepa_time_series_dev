import os

import hydra
import hydra.core
import numpy as np
import torch
from omegaconf import DictConfig

from src.base.model import BaseModel
from src.engines.contrastive_pretrain import (
    ContrastivePretrainEngine,
)
from src.factories.loader import get_loaders
from src.factories.logger import LoggerFactory
from src.factories.model import ModelFactory


def set_seed(seed: int):
    """
    设置随机种子。
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(
        os.getcwd(),
        "configs",
    ),
    config_name="default",
)
def main(cfg: DictConfig):
    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    task_kwargs = cfg["task"]

    seed = int(cfg["seed"])

    set_seed(seed)

    his_len = int(
        task_kwargs["his_len"]
    )

    pred_len = int(
        task_kwargs["pred_len"]
    )

    device_id = int(
        exp_kwargs["device_id"]
    )

    device = (
        f"cuda:{device_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    meta_path = cfg["meta_path"]

    # ================================================================
    # 1. 构建数据加载器
    # ================================================================
    (
        loaders,
        adj_mtx,
        norm_pipeline,
        num_features,
    ) = get_loaders(
        meta_path,
        dataset_kwargs,
        exp_kwargs,
        his_len,
        pred_len,
        device,
    )

    node_num = adj_mtx.shape[0]

    # ================================================================
    # 2. 构建模型参数
    # ================================================================
    ext_kwargs = dict(
        model_kwargs,
        **task_kwargs,
        **{
            "node_num": node_num,
            "input_dim": (
                num_features + 1
            ),
            "output_dim": 1,
        },
    )

    model_name = (
        model_kwargs["_target_"]
        .split(".")[-1]
    )

    model: BaseModel = (
        ModelFactory.create_model(
            model_name=model_name,
            ext_kwargs=ext_kwargs,
            adj_mx=adj_mtx,
            device=device,
        )
    )

    # ================================================================
    # 3. 日志目录
    # ================================================================
    log_folder = (
        hydra.core
        .hydra_config
        .HydraConfig
        .get()
        .runtime
        .output_dir
    )

    logger = LoggerFactory.create_logger(
        log_folder,
        mode="contrastive_pretrain",
    )

    logger.info(
        f"Contrastive model: {model_name}"
    )

    logger.info(
        f"Contrastive dataset: "
        f"{dataset_kwargs['name']}"
    )

    logger.info(
        f"Node number: {node_num}"
    )

    logger.info(
        f"History length: {his_len}"
    )

    logger.info(
        f"Prediction length: {pred_len}"
    )

    # ================================================================
    # 4. 构建对比学习 Engine
    # ================================================================
    engine = ContrastivePretrainEngine(
        device=device,
        model=model,
        logger=logger,
        save_path=log_folder,
        scalar=norm_pipeline,
        train_loader=(
            loaders["train_loader"]
        ),
        config=exp_kwargs,
        model_name=model_name,
    )

    engine.run()


if __name__ == "__main__":
    main()