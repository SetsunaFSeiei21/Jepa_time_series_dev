#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# ================================
# Spectral JEPA smoke test
# ================================
#
# Scope:
#   dataset: pulse
#   model: patchtst_spectral
#   seed: 0
#
# Training:
#   1 epoch pretraining
#   1 epoch fine-tuning
#
# Purpose:
#   1. Check model/config import.
#   2. Check spectral context C update.
#   3. Check checkpoint saving.
#   4. Check pretrained checkpoint loading.
#   5. Check fine-tuning forward/backward.
# ================================

DATASETS=(pulse)
TASKS=(time-series)
SEEDS=(0)
MODELS=(patchtst_spectral)

PRETRAIN_EPOCHS=1
FINETUNE_EPOCHS=1

NUM_WORKERS=1

DATE_TAG=$(date +"%Y-%m-%d-%H%M%S")

# 与正式实验目录分开，防止覆盖
ROOT_DIR="time-series-spectral-smoke/${DATE_TAG}"

RUNNER="scripts/test_run/scut_slurm_run.sh"


# ================================
# Basic checks
# ================================

if [[ ! -f "${RUNNER}" ]]; then
  echo "[ERROR] Slurm runner does not exist:"
  echo "        ${RUNNER}"
  exit 1
fi

if [[ ! -f "src/models/time-series/patchtst_spectral.py" ]]; then
  echo "[ERROR] Spectral PatchTST implementation does not exist:"
  echo "        src/models/time-series/patchtst_spectral.py"
  exit 1
fi

# 先检查 Python 语法，避免提交任务后才发现语法错误
python -m py_compile \
  src/models/time-series/patchtst_spectral.py \
  src/engines/jepa_pretrain.py

echo "[INFO] Python syntax check passed."


# ================================
# Submit smoke-test jobs
# ================================

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  MODEL_CONFIG="configs/model/${model}.yaml"

  if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "[ERROR] Model config does not exist:"
    echo "        ${MODEL_CONFIG}"
    exit 1
  fi

  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for task in "${TASKS[@]}"; do

        #################################
        # S1. Spectral JEPA pretrain
        #     1 epoch
        #################################

        PRETRAIN_DIR="${MODEL_ROOT_DIR}/pretrain${PRETRAIN_EPOCHS}/${dataset}/${task}/seed_${seed}"

        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_${PRETRAIN_EPOCHS}.pth"

        echo "============================================================"
        echo "[SMOKE SUBMIT] Spectral JEPA pretraining"
        echo "model=${model}"
        echo "dataset=${dataset}"
        echo "task=${task}"
        echo "seed=${seed}"
        echo "epochs=${PRETRAIN_EPOCHS}"
        echo "output=${PRETRAIN_DIR}"
        echo "checkpoint=${PRETRAIN_CKPT}"
        echo "============================================================"

        PRETRAIN_JOB_ID=$(sbatch --parsable \
          -J "smk-spt-${model_tag}-${dataset}-s${seed}" \
          -o "hpc_logs/smk-spt-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/smk-spt-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.pretrain \
            dataset="${dataset}" \
            model="${model}" \
            task="${task}" \
            exp=jepa_pretrain \
            seed="${seed}" \
            exp.max_epochs="${PRETRAIN_EPOCHS}" \
            exp.save_freq=1 \
            exp.num_workers="${NUM_WORKERS}" \
            hydra.run.dir="${PRETRAIN_DIR}")

        # 部分 Slurm 集群可能返回：
        #   123456;cluster_name
        PRETRAIN_JOB_ID="${PRETRAIN_JOB_ID%%;*}"

        if [[ -z "${PRETRAIN_JOB_ID}" ]]; then
          echo "[ERROR] Failed to obtain pretraining job ID."
          exit 1
        fi

        echo "[INFO] Pretraining job ID: ${PRETRAIN_JOB_ID}"


        #################################
        # S2. Fine-tune
        #     1 epoch after pretraining
        #################################

        PT_FT_DIR="${MODEL_ROOT_DIR}/pretrain${PRETRAIN_EPOCHS}_finetune${FINETUNE_EPOCHS}/${dataset}/${task}/seed_${seed}"

        echo "============================================================"
        echo "[SMOKE SUBMIT] Fine-tuning after Spectral JEPA"
        echo "model=${model}"
        echo "dataset=${dataset}"
        echo "task=${task}"
        echo "seed=${seed}"
        echo "epochs=${FINETUNE_EPOCHS}"
        echo "dependency=${PRETRAIN_JOB_ID}"
        echo "checkpoint=${PRETRAIN_CKPT}"
        echo "output=${PT_FT_DIR}"
        echo "============================================================"

        FINETUNE_JOB_ID=$(sbatch --parsable \
          --dependency="afterok:${PRETRAIN_JOB_ID}" \
          -J "smk-sft-${model_tag}-${dataset}-s${seed}" \
          -o "hpc_logs/smk-sft-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/smk-sft-${model_tag}-${dataset}-${task}-s${seed}.err" \
          "${RUNNER}" \
          python -m src.experiments.finetune \
            dataset="${dataset}" \
            model="${model}" \
            task="${task}" \
            exp=jepa_finetune \
            seed="${seed}" \
            exp.max_epochs="${FINETUNE_EPOCHS}" \
            exp.num_workers="${NUM_WORKERS}" \
            exp.pretrain_ckpt="${PRETRAIN_CKPT}" \
            hydra.run.dir="${PT_FT_DIR}")

        FINETUNE_JOB_ID="${FINETUNE_JOB_ID%%;*}"

        echo "[INFO] Fine-tuning job ID: ${FINETUNE_JOB_ID}"
        echo "[INFO] Fine-tuning depends on: ${PRETRAIN_JOB_ID}"

      done
    done
  done
done


echo
echo "============================================================"
echo "[INFO] Spectral JEPA smoke-test jobs submitted."
echo "[INFO] Result root: ${ROOT_DIR}"
echo
echo "[INFO] Check queue:"
echo "       squeue -u \$USER"
echo
echo "[INFO] Check pretraining log:"
echo "       tail -f hpc_logs/smk-spt-patchtst-spectral-pulse-time-series-s0.out"
echo
echo "[INFO] Check pretraining errors:"
echo "       tail -f hpc_logs/smk-spt-patchtst-spectral-pulse-time-series-s0.err"
echo
echo "[INFO] Check fine-tuning log:"
echo "       tail -f hpc_logs/smk-sft-patchtst-spectral-pulse-time-series-s0.out"
echo "============================================================"