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

set -Eeuo pipefail
set -x

trap 'echo "[ERROR] Job failed at line ${LINENO} with exit code $?" >&2' ERR

PROJECT_ROOT="/public/home/202411094934/python_project/Spatio-Temporal-Library"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ================================
# Debug / stability flags
# ================================
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_SHOW_CPP_STACKTRACES=1

# 这个会让 CUDA 报错同步暴露，但会变慢。
# Debug 阶段建议打开；确认稳定后可以注释掉。
export CUDA_LAUNCH_BLOCKING=1

# 尝试允许 core dump。集群可能会禁止，所以失败也不要让作业退出。
ulimit -c unlimited || true

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

echo "========== ENV =========="
which python
python --version
nvidia-smi || true
free -h || true
ulimit -a || true

echo "========== RUN =========="

/usr/bin/time -v "$@"