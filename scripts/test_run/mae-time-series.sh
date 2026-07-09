#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

DATASETS=(etth1 ettm1 electricity weather)
TASKS=(time-series)
SEEDS=(0 1 2)
MODELS=(patchtst crossformer fedformer)

NUM_WORKERS=8

DATE_TAG=$(date +"%Y-%m-%d")
ROOT_DIR="mae-time-series/${DATE_TAG}"
RUNNER="scripts/test_run/scut_slurm_run.sh"

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do

        #################################
        # A. MAE pretrain 10
        #################################
        PRETRAIN_DIR="${MODEL_ROOT_DIR}/mae_pretrain10/${dataset}/${task}/seed_${seed}"
        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/mae_pretrain_epoch_10.pth"

        PRETRAIN_JOB_ID=$(sbatch --parsable \
          -J "mae10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/mae10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/mae10-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.mae_pretrain \
            dataset=${dataset} \
            model=${model} \
            task=${task} \
            exp=mae_pretrain \
            seed=${seed} \
            exp.max_epochs=10 \
            exp.save_freq=2 \
            exp.num_workers=${NUM_WORKERS} \
            hydra.run.dir=${PRETRAIN_DIR})

        #################################
        # B. finetune 10 after MAE pretrain
        #################################
        MAE10_FT10_DIR="${MODEL_ROOT_DIR}/mae_pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

        sbatch \
          --dependency=afterok:${PRETRAIN_JOB_ID} \
          -J "mae10ft10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/mae10ft10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/mae10ft10-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.finetune \
            dataset=${dataset} \
            model=${model} \
            task=${task} \
            exp=jepa_finetune \
            seed=${seed} \
            exp.max_epochs=10 \
            exp.num_workers=${NUM_WORKERS} \
            exp.pretrain_ckpt=${PRETRAIN_CKPT} \
            hydra.run.dir=${MAE10_FT10_DIR}

      done
    done
  done
done