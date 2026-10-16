#!/bin/bash
# Runs one command inside a Slurm job with the project's environment:
#   sbatch "${SBATCH_ARGS[@]}" scripts/hpc/slurm_job.sh python -m src.experiments.main ...
set -euo pipefail
source scripts/hpc/cluster.sh
if [ -n "$MODULES" ]; then module purge; module load $MODULES; fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
echo "job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "$@"
