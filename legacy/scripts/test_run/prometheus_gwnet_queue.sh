#!/bin/bash
# Sequential GWNet JEPA-repeat queue for the single-GPU prometheus server.
#
# Same protocol as jepa-repeat.sh (ft20 from scratch / pt10 / pt10+ft10),
# but runs jobs one after another instead of via sbatch. Finished jobs are
# skipped, so the script can simply be re-launched after an interruption.
#
# Launch detached (survives ssh / Claude session closing):
#   tmux new-session -d -s gwnet_ln "bash scripts/test_run/prometheus_gwnet_queue.sh"
# Watch:
#   tmux attach -t gwnet_ln          (detach again with Ctrl-b d)
#   tail -f jepa-repeat/gwnet-ln/queue.log
set -uo pipefail

PROJECT_ROOT="${HOME}/workspace/Time-Series-JEPA-dev"
PYTHON="${HOME}/miniconda3/envs/tsjepa/bin/python"
META_PATH="${HOME}/STFM-datasets/"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# PEMS08 first: it is ~6x faster, so a complete set of results arrives early.
DATASETS=(pems08 pems07)
SEEDS=(0 1 2)
TASKS=(test_task)
MODEL=gwnet
NUM_WORKERS=1

ROOT_DIR="${PROJECT_ROOT}/jepa-repeat/gwnet-ln"
QUEUE_LOG="${ROOT_DIR}/queue.log"
mkdir -p "${ROOT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

finetune_done() {
  # ExpertEngine writes the final best-model test metrics under the "test" key.
  [ -f "$1/training_log.json" ] && grep -q '"test"' "$1/training_log.json"
}

# run_job <run_dir> <stdout_file> <python args...>
run_job() {
  local run_dir="$1" out="$2"
  shift 2
  # Remove partial output of an interrupted attempt before restarting it.
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  log "START ${run_dir}"
  if "${PYTHON}" "$@" hydra.run.dir="${run_dir}" > "${out}" 2>&1; then
    log "DONE  ${run_dir}"
    return 0
  fi
  log "FAIL  ${run_dir} (see ${out})"
  return 1
}

log "Queue started (pid $$)"

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      COMMON=(dataset=${dataset} model=${MODEL} task=${task} seed=${seed}
              exp.num_workers=${NUM_WORKERS} meta_path=${META_PATH})

      # A. finetune 20 from scratch (w/o JEPA)
      FT20_DIR="${ROOT_DIR}/finetune20/${dataset}/${task}/seed_${seed}"
      if finetune_done "${FT20_DIR}"; then
        log "SKIP  ${FT20_DIR} (already finished)"
      else
        run_job "${FT20_DIR}" "${FT20_DIR}.out" \
          -m src.experiments.finetune "${COMMON[@]}" \
          exp=jepa_finetune exp.max_epochs=20 'exp.pretrain_ckpt=""'
      fi

      # B1. JEPA pretrain 10
      PT_DIR="${ROOT_DIR}/pretrain10/${dataset}/${task}/seed_${seed}"
      PT_CKPT="${PT_DIR}/jepa_pretrain_epoch_10.pth"
      if [ -f "${PT_CKPT}" ]; then
        log "SKIP  ${PT_DIR} (already finished)"
      else
        run_job "${PT_DIR}" "${PT_DIR}.out" \
          -m src.experiments.pretrain "${COMMON[@]}" \
          exp=jepa_pretrain exp.max_epochs=10 exp.save_freq=2
      fi

      # B2. finetune 10 after pretrain (w/ JEPA)
      PTFT_DIR="${ROOT_DIR}/pretrain10_finetune10/${dataset}/${task}/seed_${seed}"
      if finetune_done "${PTFT_DIR}"; then
        log "SKIP  ${PTFT_DIR} (already finished)"
      elif [ ! -f "${PT_CKPT}" ]; then
        log "SKIP  ${PTFT_DIR} (missing ${PT_CKPT})"
      else
        run_job "${PTFT_DIR}" "${PTFT_DIR}.out" \
          -m src.experiments.finetune "${COMMON[@]}" \
          exp=jepa_finetune exp.max_epochs=10 exp.pretrain_ckpt="${PT_CKPT}"
      fi
    done
  done
done

log "Queue finished"
"${PYTHON}" scripts/test_run/summarize_jepa_repeat.py "${ROOT_DIR}" | tee -a "${QUEUE_LOG}"
