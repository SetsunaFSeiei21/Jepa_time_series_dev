#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/202411094934/python_project/Spatio-Temporal-Library}"

RUNNER="${PROJECT_ROOT}/scripts/test_run/scut_slurm_run.sh"

MODEL="${1:-patchtst}"
DATASET="${2:-etth1}"
SEED="${3:-0}"
TASK="time-series"

cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

DATE_TAG=$(date +"%Y-%m-%d")

ROOT_DIR="contrastive-smoke/${DATE_TAG}/${MODEL}/${DATASET}/${TASK}/seed_${SEED}"

PRETRAIN_DIR="${ROOT_DIR}/pretrain1"

FINETUNE_DIR="${ROOT_DIR}/pretrain1_finetune1"

PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/contrastive_pretrain_epoch_1.pth"


echo "=========================================="
echo "[Contrastive Smoke Test]"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Pretrain dir: ${PRETRAIN_DIR}"
echo "Checkpoint: ${PRETRAIN_CKPT}"
echo "=========================================="


# ============================================================
# 1. Contrastive pretrain 1 epoch
# ============================================================

PRETRAIN_JOB_ID=$(sbatch --parsable \
    -J "ctr-smk-pt-${MODEL}-${DATASET}-s${SEED}" \
    -o "hpc_logs/ctr-smk-pt-${MODEL}-${DATASET}-s${SEED}.out" \
    -e "hpc_logs/ctr-smk-pt-${MODEL}-${DATASET}-s${SEED}.err" \
    "${RUNNER}" \
    python -m src.experiments.contrastive_pretrain \
        dataset="${DATASET}" \
        model="${MODEL}" \
        task="${TASK}" \
        exp=contrastive_pretrain \
        seed="${SEED}" \
        exp.max_epochs=1 \
        exp.save_freq=1 \
        exp.batch_size=2 \
        exp.accumulation_steps=1 \
        exp.num_workers=2 \
        exp.prefetch_factor=2 \
        hydra.run.dir="${PRETRAIN_DIR}"
)


echo "Submitted contrastive pretraining job:"
echo "  job_id=${PRETRAIN_JOB_ID}"

echo "Expected checkpoint:"
echo "  ${PRETRAIN_CKPT}"


# ============================================================
# 2. Finetune 1 epoch
# ============================================================

FINETUNE_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${PRETRAIN_JOB_ID}" \
    -J "ctr-smk-ft-${MODEL}-${DATASET}-s${SEED}" \
    -o "hpc_logs/ctr-smk-ft-${MODEL}-${DATASET}-s${SEED}.out" \
    -e "hpc_logs/ctr-smk-ft-${MODEL}-${DATASET}-s${SEED}.err" \
    "${RUNNER}" \
    python -m src.experiments.finetune \
        dataset="${DATASET}" \
        model="${MODEL}" \
        task="${TASK}" \
        exp=jepa_finetune \
        seed="${SEED}" \
        exp.max_epochs=1 \
        exp.batch_size=1 \
        exp.accumulation_steps=1 \
        exp.num_workers=2 \
        exp.prefetch_factor=2 \
        exp.pretrain_ckpt="${PRETRAIN_CKPT}" \
        hydra.run.dir="${FINETUNE_DIR}"
)


echo "Submitted finetuning job:"
echo "  job_id=${FINETUNE_JOB_ID}"

echo "Finetuning depends on:"
echo "  ${PRETRAIN_JOB_ID}"