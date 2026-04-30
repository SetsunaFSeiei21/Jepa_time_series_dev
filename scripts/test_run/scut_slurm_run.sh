#!/bin/bash
#SBATCH --partition=gpuA800
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=934104070@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source /public/home/202411094934/anaconda3/etc/profile.d/conda.sh
conda activate newbase

echo "=========================================="
echo "[SCUT Slurm Runner]"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node List: ${SLURM_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Working Dir: $(pwd)"
echo "Command: $@"
echo "=========================================="

exec "$@"