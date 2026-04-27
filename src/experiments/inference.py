import os
import argparse
import yaml
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig


import hydra.core
from src.base.engine import BaseEngine
from src.base.model import BaseModel
from src.factories.engine import get_engine
from src.factories.loader import get_loaders
from src.factories.model import get_model
from factories.logger import LoggerFactory


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(log_folder: str) -> DictConfig:
    """
    从日志目录加载Hydra配置
    Args:
        log_folder: 包含.hydra/config.yaml的日志目录路径
    Returns:
        OmegaConf配置对象
    """
    config_path = os.path.join(log_folder, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 加载YAML并转换为OmegaConf配置对象[1,2](@ref)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)


def main(log_folder: str):
    """
    主推理函数
    Args:
        log_folder: 日志目录路径，包含模型和配置
    """
    # 1. 加载训练配置[1,2](@ref)
    cfg = load_config(log_folder)

    # 2. 提取配置参数
    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    task_exp = cfg["task"]

    # 设置随机种子
    set_seed(exp_kwargs["seed"])

    # 提取关键参数
    HIS_LEN = task_exp["seq_len"]
    PRED_LEN = task_exp["pred_len"]
    RES = task_exp["resolution"]
    LABEL = task_exp["label"]
    DEVICE = f"cuda:{exp_kwargs['device_id']}" if torch.cuda.is_available() else "cpu"
    meta_path = cfg["meta_path"]

    # 3. 准备数据加载器[7](@ref)
    loaders, scaler, adj_mtx, node_mask, num_feature = get_loaders(
        meta_path, dataset_kwargs, exp_kwargs, HIS_LEN, PRED_LEN, RES, LABEL, DEVICE
    )

    # 4. 构建模型[9,10](@ref)
    N_NODE = adj_mtx.shape[0]
    N_FEAT = num_feature + 1
    ext_kwargs = {
        **model_kwargs,
        "node_num": N_NODE,
        "input_dim": N_FEAT,
        "output_dim": 1,
        "his_len": HIS_LEN,
        "pred_len": PRED_LEN,
    }

    model_name = model_kwargs["_target_"].split(".")[-1]
    model: BaseModel = get_model(model_name, ext_kwargs, adj_mtx, DEVICE)

    # 6. 初始化日志和推理引擎[4](@ref)
    logger = LoggerFactory.create_logger(log_folder, mode="inference")
    logger.info("Node Mask: %s", np.where(node_mask)[0].tolist())

    # 7. 创建并运行推理引擎[10,11](@ref)
    engine: BaseEngine = get_engine(model_name)(
        device=DEVICE,
        model=model,
        dataloader=loaders,
        scaler=scaler,
        loss_fn=lambda x, y: torch.nn.functional.l1_loss(x, y, reduction="mean"),
        exp_kwargs=exp_kwargs,
        log_dir=log_folder,
        logger=logger,
    )

    # 8. 执行推理（仅测试）[9,10](@ref)
    engine.inference()


if __name__ == "__main__":
    # 设置命令行参数解析[7](@ref)
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument(
        "--log_folder",
        type=str,
        default="outputs/Sydney/8/bike/patchtst",
        help="Log directory containing .hydra/config.yaml and model_parameters.pt",
    )

    args = parser.parse_args()

    # 处理路径中的空格（支持Linux/Windows）[6](@ref)
    log_folder = args.log_folder
    if " " in log_folder and not (
        log_folder.startswith('"') and log_folder.endswith('"')
    ):
        log_folder = f'"{log_folder}"'

    # 检查路径是否存在
    if not os.path.exists(log_folder):
        raise NotADirectoryError(f"The specified log directory {log_folder} does not exist.")

    main(log_folder)
