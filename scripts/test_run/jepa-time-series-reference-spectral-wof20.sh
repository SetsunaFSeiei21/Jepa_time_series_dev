#!/bin/bash
set -euo pipefail


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs


# ============================================================
# 2. 实验配置
# ============================================================

DATASETS=(
  etth2
  exchange_rate
)

TASKS=(
  time-series
)

SEEDS=(
  0
  1
  2
)

# 后续可以在这里继续添加其他模型。
MODELS=(
  patchtst_reference_spectral
)

NUM_WORKERS=1

PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=10


# ============================================================
# 3. 输出路径
# ============================================================

DATE_TAG=$(date +"%Y-%m-%d")

ROOT_DIR="time-series-reference-spectral/${DATE_TAG}"

RUNNER="scripts/test_run/scut_slurm_run.sh"


# ============================================================
# 4. 提交实验
# ============================================================

for seed in "${SEEDS[@]}"; do

  for dataset in "${DATASETS[@]}"; do

    for task in "${TASKS[@]}"; do

      for model in "${MODELS[@]}"; do

        # Slurm job name 中使用较短的模型标识。
        model_tag="${model}"

        echo "============================================================"
        echo "[INFO] model   = ${model}"
        echo "[INFO] dataset = ${dataset}"
        echo "[INFO] task    = ${task}"
        echo "[INFO] seed    = ${seed}"
        echo "============================================================"


        # ====================================================
        # 4.1 JEPA 预训练
        # ====================================================

        PRETRAIN_DIR="${ROOT_DIR}/${model}/pretrain${PRETRAIN_EPOCHS}/${dataset}/${task}/seed_${seed}"

        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_${PRETRAIN_EPOCHS}.pth"

        PRETRAIN_JOB_ID=$(
          sbatch --parsable \
            -J "ref-pre-${model_tag}-${dataset}-s${seed}" \
            -o "hpc_logs/ref-pre-${model_tag}-${dataset}-s${seed}.out" \
            -e "hpc_logs/ref-pre-${model_tag}-${dataset}-s${seed}.err" \
            "${RUNNER}" \
            python -m src.experiments.pretrain \
              dataset="${dataset}" \
              model="${model}" \
              task="${task}" \
              exp=jepa_reference_pretrain \
              seed="${seed}" \
              exp.max_epochs="${PRETRAIN_EPOCHS}" \
              exp.num_workers="${NUM_WORKERS}" \
              hydra.run.dir="${PRETRAIN_DIR}"
        )

        # 部分 Slurm 环境可能返回：
        #
        #     123456;cluster_name
        #
        # 这里只保留纯 job ID。
        PRETRAIN_JOB_ID="${PRETRAIN_JOB_ID%%;*}"

        echo "[INFO] Submitted pretraining job:"
        echo "       job_id = ${PRETRAIN_JOB_ID}"
        echo "       output = ${PRETRAIN_DIR}"
        echo "       ckpt   = ${PRETRAIN_CKPT}"


        # ====================================================
        # 4.2 微调
        # ====================================================

        FINETUNE_DIR="${ROOT_DIR}/${model}/pretrain${PRETRAIN_EPOCHS}_finetune${FINETUNE_EPOCHS}/${dataset}/${task}/seed_${seed}"

        FINETUNE_JOB_ID=$(
          sbatch --parsable \
            --dependency="afterok:${PRETRAIN_JOB_ID}" \
            -J "ref-ft-${model_tag}-${dataset}-s${seed}" \
            -o "hpc_logs/ref-ft-${model_tag}-${dataset}-s${seed}.out" \
            -e "hpc_logs/ref-ft-${model_tag}-${dataset}-s${seed}.err" \
            "${RUNNER}" \
            python -m src.experiments.finetune \
              dataset="${dataset}" \
              model="${model}" \
              task="${task}" \
              exp=jepa_reference_finetune \
              seed="${seed}" \
              exp.max_epochs="${FINETUNE_EPOCHS}" \
              exp.num_workers="${NUM_WORKERS}" \
              exp.pretrain_ckpt="${PRETRAIN_CKPT}" \
              hydra.run.dir="${FINETUNE_DIR}"
        )

        FINETUNE_JOB_ID="${FINETUNE_JOB_ID%%;*}"

        echo "[INFO] Submitted finetuning job:"
        echo "       job_id    = ${FINETUNE_JOB_ID}"
        echo "       dependency= afterok:${PRETRAIN_JOB_ID}"
        echo "       output    = ${FINETUNE_DIR}"
        echo

      done  # model

    done  # task

  done  # dataset

done  # seed