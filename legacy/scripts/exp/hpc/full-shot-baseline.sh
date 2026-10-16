#!/bin/bash
for iter in {0..2}; do
    for dataset in pems03 pems04 pems07 pems08 pems-bay metr-la; do
        for task in short long; do
            for model in agcrn astgcn dstagnn stgcn gwnet sttn staeformer autoformer patchtst dlinear lstm itransformer fedformer; do
                HYDRA_ARGS=(
                    dataset=${dataset}
                    task=${task}
                    model=${model} 
                    seed=${iter}
                    hydra.sweep.dir=full-shot-baseline 
                    hydra.sweep.subdir=\"${dataset}/${task}/${model}\"
                )
                
                sbatch \
                -J "baseline-${dataset}-${task}-${model}" \
                -e "hpc_logs/baseline-${dataset}-${task}-${model}.err" \
                scripts/exp/hpc/slurm_run.sh \
                python src/experiments/main.py --multirun "${HYDRA_ARGS[@]}"
            done
        done
    done
done