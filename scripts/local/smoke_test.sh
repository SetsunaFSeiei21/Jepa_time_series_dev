#!/bin/bash
# Local smoke test: unit/contract tests, then for every model one epoch of supervised training,
# of every pretraining method and of fine-tuning from the pretrained checkpoint, on a small
# graph dataset (and a time-series dataset for the time-series models). Each run is capped to
# a few batches of data by a large train stride and a single epoch.
#   bash scripts/local/smoke_test.sh [model ...]
set -euo pipefail
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-runs/smoke}"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(gwnet stgcn astgcn sttn staeformer autoformer fedformer crossformer itransformer patchtst)
fi
METHODS=(jepa mae contrastive)
TIME_SERIES_MODELS=" autoformer fedformer crossformer itransformer patchtst "
# 1 epoch on every 50th training window keeps a run to seconds while exercising the whole loop.
QUICK=(train.max_epochs=1 train.train_stride=50 train.device="$DEVICE")

"$PYTHON" -m pytest -q tests

run() {  # run <name> <entry module> <overrides...>
  local name="$1" module="$2"
  shift 2
  echo "== $name"
  "$PYTHON" -m "$module" "$@" "${QUICK[@]}" hydra.run.dir="$OUT/$name" > "$OUT/$name.log" 2>&1 \
    || { echo "FAILED: $name (see $OUT/$name.log)"; return 1; }
}

mkdir -p "$OUT"
status=0
for model in "${MODELS[@]}"; do
  settings=("pems08 short")
  if [[ "$TIME_SERIES_MODELS" == *" $model "* ]]; then settings+=("etth1 long"); fi
  for setting in "${settings[@]}"; do
    read -r dataset task <<< "$setting"
    common=(model="$model" dataset="$dataset" task="$task")
    tag="$model-$dataset-$task"
    run "$tag-scratch" src.experiments.main "${common[@]}" || status=1
    for method in "${METHODS[@]}"; do
      run "$tag-$method-pre" src.experiments.pretrain "${common[@]}" pretrain="$method" || { status=1; continue; }
      run "$tag-$method-ft" src.experiments.main "${common[@]}" \
        train.init_checkpoint="$OUT/$tag-$method-pre/pretrained_model.pt" || status=1
    done
  done
done
exit $status
