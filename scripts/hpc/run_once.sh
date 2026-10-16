#!/bin/bash
# One independent repeat of the JEPA protocol for one model / dataset / task / seed, its stages
# run in sequence:
#   1. supervised training from scratch                      (SCRATCH_EPOCHS)
#   2. self-supervised pretraining with METHOD               (PRETRAIN_EPOCHS)
#   3. supervised training from the pretrained checkpoint    (FINETUNE_EPOCHS)
# A finished stage is skipped and an unfinished one restarts in a clean directory, so a failed or
# timed-out repeat is completed by running the same command again. The micro-batch and gradient
# accumulation (effective batch 8) come from configs/batch_sizes.yaml.
#   bash scripts/hpc/run_once.sh gwnet pems08 short 0                     # in the current shell
#   sbatch "${SBATCH_ARGS[@]}" scripts/hpc/slurm_job.sh bash scripts/hpc/run_once.sh gwnet pems08 short 0
# Run from the repository root.
set -euo pipefail
source scripts/hpc/cluster.sh
if [ $# -ne 4 ]; then
  echo "usage: $0 <model> <dataset> <task> <seed>" >&2
  exit 1
fi
model="$1" dataset="$2" task="$3" seed="$4"

TAG="${TAG:-jepa-repeat}"                  # runs are written to runs/<TAG>/...
METHOD="${METHOD:-jepa}"                   # pretraining method: jepa | mae | contrastive
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-20}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

batch=$(python tools/batch_overrides.py "$model" "$dataset" "$task")
run="$model/$dataset/$task/seed_$seed"
common=(model="$model" dataset="$dataset" task="$task" seed="$seed" data_root="$DATA_ROOT" $batch)

stage() {  # stage <run dir> <file it leaves when finished> <command...>
  local dir="$1" done_file="$2"
  shift 2
  if [ -f "$dir/$done_file" ]; then
    echo "[$(date '+%F %T')] done already: $dir"
    return
  fi
  rm -rf "$dir"  # a partial run would mix its history.jsonl into the new one
  echo "[$(date '+%F %T')] start: $dir"
  "$@" hydra.run.dir="$dir"
}

stage "runs/$TAG/scratch/$run" test_metrics.json \
  python -m src.experiments.main "${common[@]}" train.max_epochs="$SCRATCH_EPOCHS"

pre_dir="runs/$TAG/$METHOD/pretrain/$run"
stage "$pre_dir" pretrained_model.pt \
  python -m src.experiments.pretrain "${common[@]}" pretrain="$METHOD" \
  train.max_epochs="$PRETRAIN_EPOCHS"

stage "runs/$TAG/$METHOD/finetune/$run" test_metrics.json \
  python -m src.experiments.main "${common[@]}" train.max_epochs="$FINETUNE_EPOCHS" \
  train.init_checkpoint="$pre_dir/pretrained_model.pt"
