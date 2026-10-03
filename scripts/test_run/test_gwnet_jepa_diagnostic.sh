#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"

cd "${PROJECT_ROOT}"

mkdir -p hpc_logs
mkdir -p diagnostics/gwnet

RUNNER="scripts/test_run/scut_slurm_run.sh"

CKPT_ROOT="${PROJECT_ROOT}/test-smoke/2026-09-14/gwnet/pretrain2/pems04/test_task/seed_1"

CKPT1="${CKPT_ROOT}/jepa_pretrain_epoch_1.pth"
CKPT2="${CKPT_ROOT}/jepa_pretrain_epoch_2.pth"

OUTPUT="${PROJECT_ROOT}/diagnostics/gwnet/pems04_test_task_seed1_epoch1_vs_epoch2.json"

sbatch \
  -J "diag-gwnet-jepa-pems04" \
  -o "hpc_logs/diag-gwnet-jepa-pems04.out" \
  -e "hpc_logs/diag-gwnet-jepa-pems04.err" \
  "${RUNNER}" \
  python scripts/debug/debug_gwnet_jepa.py \
    --dataset pems04 \
    --model gwnet \
    --task test_task \
    --exp jepa_pretrain \
    --seed 1 \
    --batch-size 1 \
    --num-workers 3 \
    --num-batches 4 \
    --sample-vectors 1024 \
    --checkpoint "${CKPT1}" \
    --checkpoint "${CKPT2}" \
    --output "${OUTPUT}"