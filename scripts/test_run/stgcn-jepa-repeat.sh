#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# smoke test 阶段先只跑一组
# DATASETS=(pems03)
# TASKS=(short)
# SEEDS=(0)

# 全量实验时再改成下面三行：
DATASETS=(pems03 pems04 pems07 pems08 pems-bay metr-la)
TASKS=(short long)
SEEDS=(0 1 2)

MODEL=stgcn_jepa
NUM_WORKERS=8

ROOT_DIR="stgcn-jepa-repeat"
RUNNER="scripts/test_run/scut_slurm_run.sh"

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do

      ############################
      # A. finetune 20 from scratch
      ############################
      FT20_DIR="${ROOT_DIR}/finetune20/${dataset}/${task}/seed_${seed}"

      sbatch \
        -J "ft20-${dataset}-${task}-s${seed}" \
        -o "hpc_logs/ft20-${dataset}-${task}-s${seed}.out" \
        -e "hpc_logs/ft20-${dataset}-${task}-s${seed}.err" \
        "${RUNNER}" \
        python -m src.experiments.finetune \
          dataset=${dataset} \
          model=${MODEL} \
          task=${task} \
          exp=jepa_finetune \
          seed=${seed} \
          exp.max_epochs=20 \
          exp.num_workers=${NUM_WORKERS} \
          'exp.pretrain_ckpt=""' \
          hydra.run.dir=${FT20_DIR}

      #################################
      # B1. pretrain 10
      #################################
      PRETRAIN_DIR="${ROOT_DIR}/pretrain10/${dataset}/${task}/seed_${seed}"
      PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_10.pth"

      PRETRAIN_JOB_ID=$(sbatch --parsable \
        -J "pt10-${dataset}-${task}-s${seed}" \
        -o "hpc_logs/pt10-${dataset}-${task}-s${seed}.out" \
        -e "hpc_logs/pt10-${dataset}-${task}-s${seed}.err" \
        "${RUNNER}" \
        python -m src.experiments.pretrain \
          dataset=${dataset} \
          model=${MODEL} \
          task=${task} \
          exp=jepa_pretrain \
          seed=${seed} \
          exp.max_epochs=10 \
          exp.save_freq=10 \
          exp.num_workers=${NUM_WORKERS} \
          hydra.run.dir=${PRETRAIN_DIR})

      #################################
      # B2. finetune 10 after pretrain
      #################################
      PT10_FT10_DIR="${ROOT_DIR}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

      sbatch \
        --dependency=afterok:${PRETRAIN_JOB_ID} \
        -J "pt10ft10-${dataset}-${task}-s${seed}" \
        -o "hpc_logs/pt10ft10-${dataset}-${task}-s${seed}.out" \
        -e "hpc_logs/pt10ft10-${dataset}-${task}-s${seed}.err" \
        "${RUNNER}" \
        python -m src.experiments.finetune \
          dataset=${dataset} \
          model=${MODEL} \
          task=${task} \
          exp=jepa_finetune \
          seed=${seed} \
          exp.max_epochs=10 \
          exp.num_workers=${NUM_WORKERS} \
          exp.pretrain_ckpt=${PRETRAIN_CKPT} \
          hydra.run.dir=${PT10_FT10_DIR}

    done
  done
done