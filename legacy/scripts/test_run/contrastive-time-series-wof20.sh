#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# ================================
# Experiment scope
# ================================

# 全部数据集：
# DATASETS=(
#   electricity
#   etth1
#   etth2
#   ettm1
#   ettm2
#   exchange_rate
#   pulse
#   weather
#   traffic
# )

DATASETS=(exchange_rate)
TASKS=(time-series)
SEEDS=(0 1 2)

# 全部模型：
# MODELS=(
#   crossformer
#   itransformer
#   autoformer
#   fedformer
#   patchtst
# )

MODELS=(fedformer)

# 对比学习真实 batch size 必须尽量大于等于 2。
# 对 Autoformer/FEDformer，batch_size=1 时可能无法构造 InfoNCE 负样本。
PRETRAIN_BATCH_SIZE=2
FINETUNE_BATCH_SIZE=1

PRETRAIN_ACCUMULATION_STEPS=1
FINETUNE_ACCUMULATION_STEPS=1

NUM_WORKERS=1
PREFETCH_FACTOR=2

DATE_TAG=$(date +"%Y-%m-%d")

# 与 JEPA 的 time-series/${DATE_TAG} 分开，避免覆盖结果。
ROOT_DIR="contrastive-time-series/${DATE_TAG}"

RUNNER="scripts/test_run/scut_slurm_run.sh"

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do

        ########################################
        # B1. Contrastive pretrain 10 epochs
        ########################################
        PRETRAIN_DIR="${MODEL_ROOT_DIR}/pretrain10/${dataset}/${task}/seed_${seed}"

        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/contrastive_pretrain_epoch_10.pth"

        echo "=========================================="
        echo "[SUBMIT] contrastive pretrain10"
        echo "model=${model}"
        echo "dataset=${dataset}"
        echo "task=${task}"
        echo "seed=${seed}"
        echo "pretrain_dir=${PRETRAIN_DIR}"
        echo "checkpoint=${PRETRAIN_CKPT}"
        echo "=========================================="

        PRETRAIN_JOB_ID=$(sbatch --parsable \
          -J "ctr-pt10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/ctr-pt10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/ctr-pt10-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.contrastive_pretrain \
            dataset="${dataset}" \
            model="${model}" \
            task="${task}" \
            exp=contrastive_pretrain \
            seed="${seed}" \
            exp.max_epochs=10 \
            exp.save_freq=2 \
            exp.batch_size=${PRETRAIN_BATCH_SIZE} \
            exp.accumulation_steps=${PRETRAIN_ACCUMULATION_STEPS} \
            exp.num_workers=${NUM_WORKERS} \
            exp.prefetch_factor=${PREFETCH_FACTOR} \
            hydra.run.dir="${PRETRAIN_DIR}"
        )

        echo "[INFO] Submitted contrastive pretraining job:"
        echo "       job_id=${PRETRAIN_JOB_ID}"

        ########################################
        # B2. Finetune 10 epochs after pretrain
        ########################################
        PT10_FT10_DIR="${MODEL_ROOT_DIR}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

        echo "=========================================="
        echo "[SUBMIT] contrastive pretrain10_finetune10"
        echo "model=${model}"
        echo "dataset=${dataset}"
        echo "task=${task}"
        echo "seed=${seed}"
        echo "dependency=${PRETRAIN_JOB_ID}"
        echo "finetune_dir=${PT10_FT10_DIR}"
        echo "checkpoint=${PRETRAIN_CKPT}"
        echo "=========================================="

        FINETUNE_JOB_ID=$(sbatch --parsable \
          --dependency="afterok:${PRETRAIN_JOB_ID}" \
          -J "ctr-pt10ft10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/ctr-pt10ft10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/ctr-pt10ft10-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.finetune \
            dataset="${dataset}" \
            model="${model}" \
            task="${task}" \
            exp=jepa_finetune \
            seed="${seed}" \
            exp.max_epochs=10 \
            exp.batch_size=${FINETUNE_BATCH_SIZE} \
            exp.accumulation_steps=${FINETUNE_ACCUMULATION_STEPS} \
            exp.num_workers=${NUM_WORKERS} \
            exp.prefetch_factor=${PREFETCH_FACTOR} \
            exp.pretrain_ckpt="${PRETRAIN_CKPT}" \
            hydra.run.dir="${PT10_FT10_DIR}"
        )

        echo "[INFO] Submitted finetuning job:"
        echo "       job_id=${FINETUNE_JOB_ID}"
        echo "       dependency=${PRETRAIN_JOB_ID}"

      done
    done
  done
done