#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

DATASETS=(
  etth2
  exchange_rate
)

TASKS=(time-series)
SEEDS=(0 1 2)

MODEL="patchtst_reference_spectral"

NUM_WORKERS=1
PRETRAIN_EPOCHS=10
FINETUNE_EPOCHS=10

DATE_TAG=$(date +"%Y-%m-%d")

ROOT_DIR="time-series-reference-spectral/${DATE_TAG}"
RUNNER="scripts/test_run/scut_slurm_run.sh"


for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do

      PRETRAIN_DIR="${ROOT_DIR}/${MODEL}/pretrain10/${dataset}/${task}/seed_${seed}"

      PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_${PRETRAIN_EPOCHS}.pth"

      PRETRAIN_JOB_ID=$(sbatch --parsable \
        -J "ref-pre-${dataset}-s${seed}" \
        -o "hpc_logs/ref-pre-${dataset}-s${seed}.out" \
        -e "hpc_logs/ref-pre-${dataset}-s${seed}.err" \
        "${RUNNER}" \
        python -m src.experiments.pretrain \
          dataset=${dataset} \
          model=${MODEL} \
          task=${task} \
          exp=jepa_reference_pretrain \
          seed=${seed} \
          exp.max_epochs=${PRETRAIN_EPOCHS} \
          exp.num_workers=${NUM_WORKERS} \
          hydra.run.dir=${PRETRAIN_DIR})

      PRETRAIN_JOB_ID="${PRETRAIN_JOB_ID%%;*}"

      FINETUNE_DIR="${ROOT_DIR}/${MODEL}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

      sbatch \
        --dependency=afterok:${PRETRAIN_JOB_ID} \
        -J "ref-ft-${dataset}-s${seed}" \
        -o "hpc_logs/ref-ft-${dataset}-s${seed}.out" \
        -e "hpc_logs/ref-ft-${dataset}-s${seed}.err" \
        "${RUNNER}" \
        python -m src.experiments.finetune \
          dataset=${dataset} \
          model=${MODEL} \
          task=${task} \
          exp=jepa_reference_finetune \
          seed=${seed} \
          exp.max_epochs=${FINETUNE_EPOCHS} \
          exp.num_workers=${NUM_WORKERS} \
          exp.pretrain_ckpt=${PRETRAIN_CKPT} \
          hydra.run.dir=${FINETUNE_DIR}

    done
  done
done