#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# ================================
# Run mode
# ================================
# all      : 跑 finetune20 + pretrain10 + pretrain10_finetune10
# ft20     : 只跑 finetune20 from scratch
# pt10ft10 : 只跑 pretrain10 + pretrain10_finetune10
RUN_MODE="${1:-all}"

case "${RUN_MODE}" in
  all|ft20|pt10ft10)
    ;;
  *)
    echo "[ERROR] Unknown RUN_MODE: ${RUN_MODE}"
    echo "Usage:"
    echo "  bash $0 all       # 跑全部实验"
    echo "  bash $0 ft20      # 只跑 finetune20"
    echo "  bash $0 pt10ft10  # 只跑 pretrain10 + finetune10"
    exit 1
    ;;
esac

echo "[INFO] RUN_MODE = ${RUN_MODE}"

# ================================
# Experiment scope time-series-datasets
# ================================

# Smoke test example:
# DATASETS=(electricity etth1 etth2 ettm1 ettm2 exchange_rate pulse weather traffic)
DATASETS=(etth2 exchange_rate)
TASKS=(time-series)
SEEDS=(0 1 2)

# MODELS=(crossformer itransformer autoformer fedformer patchtst)
# crossformer, itransformer, autoformer, patchtst has completed
MODELS=(crossformer)

NUM_WORKERS=1

DATE_TAG=$(date +"%Y-%m-%d")
ROOT_DIR="time-series-result/${DATE_TAG}"
RUNNER="scripts/test_run/scut_slurm_run.sh"

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do

        ############################
        # A. finetune 20 from scratch
        ############################
        if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "ft20" ]]; then
          FT20_DIR="${MODEL_ROOT_DIR}/finetune20/${dataset}/${task}/seed_${seed}"

          echo "[SUBMIT] finetune20 | model=${model} dataset=${dataset} task=${task} seed=${seed}"

          sbatch \
            -J "ft20-${model_tag}-${dataset}-${task}-s${seed}" \
            -o "hpc_logs/ft20-${model_tag}-${dataset}-${task}-s${seed}.out" \
            -e "hpc_logs/ft20-${model_tag}-${dataset}-${task}-s${seed}.err" \
            "${RUNNER}" \
            python -m src.experiments.finetune \
              dataset=${dataset} \
              model=${model} \
              task=${task} \
              exp=jepa_finetune \
              seed=${seed} \
              exp.max_epochs=20 \
              exp.num_workers=${NUM_WORKERS} \
              'exp.pretrain_ckpt=""' \
              hydra.run.dir=${FT20_DIR}
        fi

        #################################
        # B1/B2. pretrain 10 + finetune 10
        #################################
        if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "pt10ft10" ]]; then

          #################################
          # B1. pretrain 10
          #################################
          PRETRAIN_DIR="${MODEL_ROOT_DIR}/pretrain10/${dataset}/${task}/seed_${seed}"
          PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_10.pth"

          echo "[SUBMIT] pretrain10 | model=${model} dataset=${dataset} task=${task} seed=${seed}"

          PRETRAIN_JOB_ID=$(sbatch --parsable \
            -J "pt10-${model_tag}-${dataset}-${task}-s${seed}" \
            -o "hpc_logs/pt10-${model_tag}-${dataset}-${task}-s${seed}.out" \
            -e "hpc_logs/pt10-${model_tag}-${dataset}-${task}-s${seed}.err" \
            "${RUNNER}" \
            python -m src.experiments.pretrain \
              dataset=${dataset} \
              model=${model} \
              task=${task} \
              exp=jepa_pretrain \
              seed=${seed} \
              exp.max_epochs=10 \
              exp.save_freq=2 \
              exp.num_workers=${NUM_WORKERS} \
              hydra.run.dir=${PRETRAIN_DIR})

          #################################
          # B2. finetune 10 after pretrain
          #################################
          PT10_FT10_DIR="${MODEL_ROOT_DIR}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

          echo "[SUBMIT] pretrain10_finetune10 | model=${model} dataset=${dataset} task=${task} seed=${seed} dependency=${PRETRAIN_JOB_ID}"

          sbatch \
            --dependency=afterok:${PRETRAIN_JOB_ID} \
            -J "pt10ft10-${model_tag}-${dataset}-${task}-s${seed}" \
            -o "hpc_logs/pt10ft10-${model_tag}-${dataset}-${task}-s${seed}.out" \
            -e "hpc_logs/pt10ft10-${model_tag}-${dataset}-${task}-s${seed}.err" \
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
              hydra.run.dir=${PT10_FT10_DIR}

        fi

      done
    done
  done
done