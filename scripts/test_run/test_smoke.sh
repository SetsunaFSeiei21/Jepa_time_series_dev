#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# ================================
# Smoke test scope: STFM-datasets
# ================================
# DATASETS=(pems07)
# TASKS=(test_task)
# SEEDS=(1)
# MODELS=(astgcn)

# ================================
# Smoke test scope: time-series-datasets
# ================================
# DATASETS=(etth1 etth2 ettm1 ettm2 illness electricity pulse traffic weather exchange_rate)
DATASETS=(etth1)
TASKS=(test_task)
SEEDS=(1)
# MODELS=(autoformer fedformer crossformer itransformer patchtst)
MODELS=(itransformer)

NUM_WORKERS=8
BATCH_SIZE=1
PREFETCH_FACTOR=2

DATE_TAG=$(date +"%Y-%m-%d")
ROOT_DIR="test-smoke/${DATE_TAG}"
RUNNER="scripts/test_run/scut_slurm_run.sh"

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do

        echo "Submitting smoke jobs: model=${model}, dataset=${dataset}, task=${task}, seed=${seed}"

        ############################
        # A. smoke finetune from scratch
        ############################
        FT1_DIR="${MODEL_ROOT_DIR}/finetune1/${dataset}/${task}/seed_${seed}"

        sbatch \
          -J "smk-ft1-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/smk-ft1-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/smk-ft1-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.finetune \
            dataset=${dataset} \
            model=${model} \
            task=${task} \
            exp=jepa_finetune \
            seed=${seed} \
            exp.max_epochs=1 \
            exp.batch_size=${BATCH_SIZE} \
            exp.num_workers=${NUM_WORKERS} \
            exp.prefetch_factor=${PREFETCH_FACTOR} \
            'exp.pretrain_ckpt=""' \
            hydra.run.dir=${FT1_DIR}

        ############################
        # B1. smoke pretrain
        ############################
        PRETRAIN_DIR="${MODEL_ROOT_DIR}/pretrain1/${dataset}/${task}/seed_${seed}"
        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_1.pth"

        PRETRAIN_JOB_ID=$(sbatch --parsable \
          -J "smk-pt1-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/smk-pt1-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/smk-pt1-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.pretrain \
            dataset=${dataset} \
            model=${model} \
            task=${task} \
            exp=jepa_pretrain \
            seed=${seed} \
            exp.max_epochs=1 \
            exp.save_freq=1 \
            exp.batch_size=${BATCH_SIZE} \
            exp.num_workers=${NUM_WORKERS} \
            exp.prefetch_factor=${PREFETCH_FACTOR} \
            hydra.run.dir=${PRETRAIN_DIR})

        echo "Submitted pretrain job ${PRETRAIN_JOB_ID}; expected checkpoint: ${PRETRAIN_CKPT}"

        ############################
        # B2. smoke finetune from pretrain
        ############################
        PT1_FT1_DIR="${MODEL_ROOT_DIR}/pretrain1_finetune1/${dataset}/${task}/seed_${seed}"

        sbatch \
          --dependency=afterok:${PRETRAIN_JOB_ID} \
          -J "smk-pt1ft1-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/smk-pt1ft1-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/smk-pt1ft1-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.finetune \
            dataset=${dataset} \
            model=${model} \
            task=${task} \
            exp=jepa_finetune \
            seed=${seed} \
            exp.max_epochs=1 \
            exp.batch_size=${BATCH_SIZE} \
            exp.num_workers=${NUM_WORKERS} \
            exp.prefetch_factor=${PREFETCH_FACTOR} \
            exp.pretrain_ckpt=${PRETRAIN_CKPT} \
            hydra.run.dir=${PT1_FT1_DIR}

      done
    done
  done
done