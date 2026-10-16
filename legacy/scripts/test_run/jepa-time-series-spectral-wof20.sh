#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"
cd "${PROJECT_ROOT}"

mkdir -p hpc_logs

# ================================
# Experiment scope time-series-datasets
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

DATASETS=(etth2 exchange_rate)
TASKS=(time-series)
SEEDS=(0 1 2)

# 全部 Spectral 模型：
# MODELS=(
#   crossformer_spectral
#   itransformer_spectral
#   autoformer_spectral
#   fedformer_spectral
#   patchtst_spectral
# )

MODELS=(patchtst_spectral)

NUM_WORKERS=1

DATE_TAG=$(date +"%Y-%m-%d")

# 与普通 JEPA 实验目录分开
ROOT_DIR="time-series-spectral/${DATE_TAG}"

RUNNER="scripts/test_run/scut_slurm_run.sh"


for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"
  MODEL_ROOT_DIR="${ROOT_DIR}/${model}"

  # 检查对应的 Hydra 模型配置是否存在
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
        # B1. Spectral JEPA pretrain
        #     10 epochs
        #################################

        PRETRAIN_DIR="${MODEL_ROOT_DIR}/pretrain10/${dataset}/${task}/seed_${seed}"

        PRETRAIN_CKPT="${PROJECT_ROOT}/${PRETRAIN_DIR}/jepa_pretrain_epoch_10.pth"

        echo "[SUBMIT] spectral_pretrain10 | model=${model} dataset=${dataset} task=${task} seed=${seed}"

        PRETRAIN_JOB_ID=$(sbatch --parsable \
          -J "spt10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/spt10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/spt10-${model_tag}-${dataset}-${task}-s${seed}.err" \
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

        # 某些 Slurm 集群可能返回：
        # 123456;cluster_name
        # dependency 只需要前面的任务编号
        PRETRAIN_JOB_ID="${PRETRAIN_JOB_ID%%;*}"

        #################################
        # B2. Finetune 10 epochs
        #     after Spectral JEPA
        #################################

        PT10_FT10_DIR="${MODEL_ROOT_DIR}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"

        echo "[SUBMIT] spectral_pretrain10_finetune10 | model=${model} dataset=${dataset} task=${task} seed=${seed} dependency=${PRETRAIN_JOB_ID}"

        sbatch \
          --dependency=afterok:${PRETRAIN_JOB_ID} \
          -J "spt10ft10-${model_tag}-${dataset}-${task}-s${seed}" \
          -o "hpc_logs/spt10ft10-${model_tag}-${dataset}-${task}-s${seed}.out" \
          -e "hpc_logs/spt10ft10-${model_tag}-${dataset}-${task}-s${seed}.err" \
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

      done
    done
  done
done