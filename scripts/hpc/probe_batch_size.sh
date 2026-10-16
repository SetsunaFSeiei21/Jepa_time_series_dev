#!/bin/bash
# Measures on the cluster GPU the largest micro-batch of every model / dataset / task and writes
# configs/batch_sizes.yaml, which scripts/hpc/run_once.sh reads.
#   bash scripts/hpc/probe_batch_size.sh
set -euo pipefail
source scripts/hpc/cluster.sh
mkdir -p hpc_logs
MODELS=(gwnet stgcn astgcn sttn staeformer autoformer fedformer crossformer itransformer patchtst)
TIME_SERIES_MODELS=(autoformer fedformer crossformer itransformer patchtst)
GRAPH_DATASETS=(pems03 pems04 pems07 pems08 pems-bay metr-la)
TIME_SERIES_DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
TASKS=(short long long192)
probe="python tools/probe_batch_size.py --data-root $DATA_ROOT --tasks ${TASKS[*]} --out configs/batch_sizes.yaml"
sbatch "${SBATCH_ARGS[@]}" -J probe-batch-size -o hpc_logs/probe-batch-size.out \
  scripts/hpc/slurm_job.sh bash -c "$probe --models ${MODELS[*]} --datasets ${GRAPH_DATASETS[*]} && \
    $probe --models ${TIME_SERIES_MODELS[*]} --datasets ${TIME_SERIES_DATASETS[*]}"
