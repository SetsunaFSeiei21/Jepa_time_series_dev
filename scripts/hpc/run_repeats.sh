#!/bin/bash
# Five independent repeats (seeds 0-4) of the JEPA protocol: submits one Slurm job running
# scripts/hpc/run_once.sh for every model / dataset / task / seed. Skipped: configurations
# without a micro-batch in configs/batch_sizes.yaml, repeats whose fine-tuning has finished, and
# repeats with a job of the same name still queued or running, so the script can be resubmitted
# after failures. Graph datasets run with every model, time-series datasets with the
# time-series models only.
#   bash scripts/hpc/run_repeats.sh
#   MODELS="gwnet stgcn" TASKS=short DRY_RUN=1 bash scripts/hpc/run_repeats.sh   # subset, print only
# TAG, METHOD and the epoch counts are passed on to run_once.sh (see there).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/cluster.sh

read -ra SEEDS <<< "${SEEDS:-0 1 2 3 4}"
read -ra MODELS <<< "${MODELS:-gwnet stgcn astgcn sttn staeformer autoformer fedformer crossformer itransformer patchtst}"
read -ra GRAPH_DATASETS <<< "${GRAPH_DATASETS:-pems03 pems04 pems07 pems08 pems-bay metr-la}"
read -ra TIME_SERIES_DATASETS <<< "${TIME_SERIES_DATASETS:-etth1 etth2 ettm1 ettm2 weather electricity traffic}"
read -ra TASKS <<< "${TASKS:-short long}"
TIME_SERIES_MODELS=" autoformer fedformer crossformer itransformer patchtst "
DRY_RUN="${DRY_RUN:-0}"                   # 1: print the sbatch commands without submitting
export TAG="${TAG:-jepa-repeat}" METHOD="${METHOD:-jepa}"
python="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"

mkdir -p hpc_logs
submitted=0
for model in "${MODELS[@]}"; do
  datasets=("${GRAPH_DATASETS[@]}")
  if [[ "$TIME_SERIES_MODELS" == *" $model "* ]]; then datasets+=("${TIME_SERIES_DATASETS[@]}"); fi
  for dataset in "${datasets[@]}"; do
    for task in "${TASKS[@]}"; do
      if ! "$python" tools/batch_overrides.py "$model" "$dataset" "$task" >/dev/null; then
        echo "skipping $model/$dataset/$task: no micro-batch in configs/batch_sizes.yaml" >&2
        continue
      fi
      for seed in "${SEEDS[@]}"; do
        run="$model/$dataset/$task/seed_$seed"
        if [ -f "runs/$TAG/$METHOD/finetune/$run/test_metrics.json" ] &&
           [ -f "runs/$TAG/scratch/$run/test_metrics.json" ]; then
          continue
        fi
        name="$TAG-$METHOD-${run//\//-}"
        if [ "$DRY_RUN" != 1 ] && [ -n "$(squeue -h -u "$USER" -n "$name" -o %i)" ]; then
          echo "skipping $name: already queued or running" >&2
          continue
        fi
        command=(sbatch "${SBATCH_ARGS[@]}" -J "$name" -o "hpc_logs/$name.out"
                 scripts/hpc/slurm_job.sh bash scripts/hpc/run_once.sh "$model" "$dataset" "$task" "$seed")
        if [ "$DRY_RUN" = 1 ]; then echo "${command[*]}"; else "${command[@]}"; fi
        submitted=$((submitted + 1))
      done
    done
  done
done
echo "$submitted jobs $([ "$DRY_RUN" = 1 ] && echo "to submit" || echo "submitted")"
