# Cluster settings shared by the Slurm scripts (SCUT cluster, A800 80 GB partition).
# Every job requests one GPU; micro-batches come from configs/batch_sizes.yaml.
PARTITION="${PARTITION:-ex01A800}"
ACCOUNT="${ACCOUNT:-b_phzhwu}"           # leave empty if the cluster has no accounts
GPU_GRES="${GPU_GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-80G}"
TIME_LIMIT="${TIME_LIMIT:-10-00:00:00}"
MAIL_USER="${MAIL_USER:-}"               # set to be mailed when a job fails
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
CONDA_ENV="${CONDA_ENV:-newbase}"        # environment with torch, hydra-core, einops, pyyaml
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}" # unified-format datasets (tools/convert_datasets.py)
MODULES="${MODULES:-}"                   # e.g. "cuda/12.4"; loaded with `module load`

# sbatch options for one GPU job; use as: sbatch "${SBATCH_ARGS[@]}" -J name ...
SBATCH_ARGS=(--partition="$PARTITION" --gres="$GPU_GRES" --cpus-per-task="$CPUS_PER_TASK"
             --mem="$MEMORY" --time="$TIME_LIMIT" --nodes=1 --ntasks=1)
if [ -n "$ACCOUNT" ]; then SBATCH_ARGS+=(--account="$ACCOUNT"); fi
if [ -n "$MAIL_USER" ]; then SBATCH_ARGS+=(--mail-user="$MAIL_USER" --mail-type=FAIL); fi
