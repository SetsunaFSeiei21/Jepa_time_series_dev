import os
import json
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="要处理的实验日期文件夹，例如 2026-05-17",
    )
    return parser.parse_args()


args = parse_args()
PROCESS_DATE = args.date

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# 输入目录：当前脚本目录 / 日期
experiment_root_dir = os.path.join(SCRIPT_DIR, PROCESS_DATE)

# 输出目录：../data_result/日期
data_result_dir = os.path.join(SCRIPT_DIR, "..", "data_result", PROCESS_DATE)
os.makedirs(data_result_dir, exist_ok=True)


MODEL_NAME = [
    "autoformer",
    "crossformer",
    "fedformer",
    "itransformer",
    "patchtst",
]

BASE_MISSION_NAME = "finetune20"
TARGET_MISSION_NAME = "pretrain10_finetune10"

DATASET_LST = [
    "electricity",
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "exchange_rate",
    "pulse",
    "weather",
    "traffic",
]

# 你的真实目录结构现在是：
# dataset/time-series/seed_0/training_log.json
MISSION_TYPE = ["time-series"]

SEED_NUMBER = 3

METRIC_NAMES = ["MAE", "RMSE"]

# finetune20 读取 epoch_logs["20"]
# pretrain10_finetune10 读取 epoch_logs["10"]
MISSION_EPOCH_KEY = {
    BASE_MISSION_NAME: "20",
    TARGET_MISSION_NAME: "10",
}


def nan_metrics():
    return {metric_name: np.nan for metric_name in METRIC_NAMES}


def safe_get_nested(data, keys):
    """
    安全读取嵌套 dict。

    例如：
        safe_get_nested(data, ["epoch_logs", "20", "validation", "metrics"])
    """
    cur = data

    for key in keys:
        if not isinstance(cur, dict):
            return None

        if key not in cur:
            return None

        cur = cur[key]

    return cur


def extract_metrics(result_data, mission_name):
    """
    从 training_log.json 中提取指标。

    finetune20:
        epoch_logs["20"]["validation"]["metrics"]

    pretrain10_finetune10:
        epoch_logs["10"]["validation"]["metrics"]

    如果缺少 20 / 10 key，或者缺少 validation / metrics / MAE / RMSE，
    直接返回 np.nan。
    """
    epoch_key = MISSION_EPOCH_KEY[mission_name]

    metrics = safe_get_nested(
        result_data,
        ["epoch_logs", epoch_key, "validation", "metrics"],
    )

    if not isinstance(metrics, dict):
        return nan_metrics()

    output_metrics = {}

    for metric_name in METRIC_NAMES:
        try:
            output_metrics[metric_name] = float(metrics[metric_name])
        except (KeyError, TypeError, ValueError):
            output_metrics[metric_name] = np.nan

    return output_metrics


def read_metrics_or_nan(result_path, mission_name):
    """
    缺文件、JSON 读取失败、缺指定 key，都返回 np.nan。
    """
    if not os.path.exists(result_path):
        print(f"[WARN] Missing result file, marked as NaN: {result_path}")
        return nan_metrics()

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON, marked as NaN: {result_path}, error={e}")
        return nan_metrics()

    metrics = extract_metrics(result_data, mission_name)

    if any(np.isnan(metrics[metric_name]) for metric_name in METRIC_NAMES):
        print(f"[WARN] Missing metric/key, marked as NaN: {result_path}")

    return metrics


def mean_std_str(values, digits=4):
    """
    只要 values 中存在 NaN，最终 mean±std 直接标注为 NaN。

    注意：
        np.std 默认是总体标准差，即 ddof=0。
    """
    values = np.array(values, dtype=float)

    if values.size == 0 or np.isnan(values).any():
        return "NaN"

    return f"{np.mean(values):.{digits}f}±{np.std(values):.{digits}f}"


def mean_value(values):
    """
    只要 values 中存在 NaN，均值返回 np.nan。
    """
    values = np.array(values, dtype=float)

    if values.size == 0 or np.isnan(values).any():
        return np.nan

    return float(np.mean(values))


def absolute_improve(base_values, target_values, digits=4):
    """
    MAE/RMSE 越低越好：

    绝对提升 = 没用 JEPA 的均值 - 用了 JEPA 的均值。

    正数表示用了 JEPA 后指标下降，即效果更好；
    负数表示用了 JEPA 后指标上升，即效果更差。

    只要 base 或 target 存在 NaN，最终 absolute improve 也标注为 NaN。
    """
    base_mean = mean_value(base_values)
    target_mean = mean_value(target_values)

    if np.isnan(base_mean) or np.isnan(target_mean):
        return "NaN"

    improve = base_mean - target_mean
    return f"{improve:.{digits}f}"


def improve_percent(base_values, target_values, digits=2):
    """
    MAE/RMSE 越低越好：

    相对提升 = (没用 JEPA 的均值 - 用了 JEPA 的均值) / 没用 JEPA 的均值 * 100%

    正数表示 pretrain10_finetune10 比 finetune20 更好；
    负数表示 pretrain10_finetune10 更差。

    只要 base 或 target 存在 NaN，最终 improve 也标注为 NaN。
    """
    base_mean = mean_value(base_values)
    target_mean = mean_value(target_values)

    if np.isnan(base_mean) or np.isnan(target_mean):
        return "NaN"

    if base_mean == 0:
        return "NaN"

    improve = (base_mean - target_mean) / base_mean * 100
    return f"{improve:.{digits}f}%"


print(f"[INFO] Input dir : {experiment_root_dir}")
print(f"[INFO] Output dir: {data_result_dir}")


# 输出总表：
# 一级 index: 模型
# 二级 index:
#   w/o JEPA
#   w/ JEPA
#   Abs. Imp.
#   Rel. Imp.
#
# 一级 columns: 数据集
# 二级 columns: MAE / RMSE
save_path = os.path.join(data_result_dir, "summary.xlsx")

with pd.ExcelWriter(save_path) as writer:
    for task in MISSION_TYPE:
        table_rows = []
        index_tuples = []

        columns = pd.MultiIndex.from_product(
            [DATASET_LST, METRIC_NAMES],
            names=["Dataset", "Metric"],
        )

        for model in MODEL_NAME:
            base_row = {}
            target_row = {}
            absolute_improve_row = {}
            relative_improve_row = {}

            for dataset in DATASET_LST:
                # 现在路径是：
                # experiment_root_dir/model/finetune20/dataset/time-series/seed_x/training_log.json
                base_dir = os.path.join(
                    experiment_root_dir,
                    model,
                    BASE_MISSION_NAME,
                    dataset,
                    task,
                )

                # 现在路径是：
                # experiment_root_dir/model/pretrain10_finetune10/dataset/time-series/seed_x/training_log.json
                target_dir = os.path.join(
                    experiment_root_dir,
                    model,
                    TARGET_MISSION_NAME,
                    dataset,
                    task,
                )

                base_metrics_lst = []
                target_metrics_lst = []

                for seed in range(SEED_NUMBER):
                    base_result_path = os.path.join(
                        base_dir,
                        f"seed_{seed}",
                        "training_log.json",
                    )

                    target_result_path = os.path.join(
                        target_dir,
                        f"seed_{seed}",
                        "training_log.json",
                    )

                    base_metrics = read_metrics_or_nan(
                        base_result_path,
                        BASE_MISSION_NAME,
                    )

                    target_metrics = read_metrics_or_nan(
                        target_result_path,
                        TARGET_MISSION_NAME,
                    )

                    base_metrics_lst.append(base_metrics)
                    target_metrics_lst.append(target_metrics)

                for metric_name in METRIC_NAMES:
                    base_values = [item[metric_name] for item in base_metrics_lst]
                    target_values = [item[metric_name] for item in target_metrics_lst]

                    base_row[(dataset, metric_name)] = mean_std_str(base_values)
                    target_row[(dataset, metric_name)] = mean_std_str(target_values)

                    absolute_improve_row[(dataset, metric_name)] = absolute_improve(
                        base_values,
                        target_values,
                    )

                    relative_improve_row[(dataset, metric_name)] = improve_percent(
                        base_values,
                        target_values,
                    )

            table_rows.extend(
                [
                    base_row,
                    target_row,
                    absolute_improve_row,
                    relative_improve_row,
                ]
            )

            index_tuples.extend(
                [
                    (model, "w/o JEPA"),
                    (model, "w/ JEPA"),
                    (model, "Abs. Imp."),
                    (model, "Rel. Imp."),
                ]
            )

        df = pd.DataFrame(
            table_rows,
            index=pd.MultiIndex.from_tuples(
                index_tuples,
                names=["Model", "Record"],
            ),
            columns=columns,
        )

        df.to_excel(
            writer,
            sheet_name=task,
            na_rep="NaN",
            merge_cells=True,
        )

print(f"Saved: {save_path}")