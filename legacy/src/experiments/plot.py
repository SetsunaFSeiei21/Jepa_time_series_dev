import os
import argparse
import yaml
import numpy as np
from matplotlib import pyplot as plt


def plot(log_folder: str):
    ground_truth = np.load(os.path.join(log_folder, "ground_truth.npy"))
    predictions = np.load(os.path.join(log_folder, "inference_outputs.npy"))
    assert (
        ground_truth.shape == predictions.shape
    ), "Shape mismatch between ground truth and predictions"
    num_nodes = ground_truth.shape[2]
    ground_truth = ground_truth[..., 0].reshape(-1, num_nodes)
    predictions = predictions[..., 0].reshape(-1, num_nodes)
    select_node = np.argmax(np.std(ground_truth, axis=0))
    plt.figure()
    plt.plot(ground_truth[:, select_node], label="Ground Truth")
    plt.plot(predictions[:, select_node], label="Predictions")
    plt.title(f"Node {select_node} - Ground Truth vs Predictions")
    plt.legend()
    plt.savefig(os.path.join(log_folder, "plot_all.png"))
    
    plt.figure()
    plt.plot(ground_truth[:24*7, select_node], label="Ground Truth")
    plt.plot(predictions[:24*7, select_node], label="Predictions")
    plt.title(f"Node {select_node} - Ground Truth vs Predictions")
    plt.legend()
    plt.savefig(os.path.join(log_folder, "plot_7day.png"))

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
        raise NotADirectoryError(
            f"The specified log directory {log_folder} does not exist."
        )

    plot(log_folder)
