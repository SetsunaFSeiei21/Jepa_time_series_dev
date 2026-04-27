#!/bin/bash
for dataset in pems03 pems04 pems07 pems08; do
    for task in short; do
        for model in agcrn astgcn dstagnn stgcn gwnet sttn staeformer autoformer patchtst dlinear lstm itransformer fedformer; do
            HYDRA_ARGS=(
                dataset=${dataset}
                task=${task}
                model=${model} 
                dataset.train_point=0.54
                exp.max_epochs=1
                hydra.sweep.dir=few-shot-baseline 
                hydra.sweep.subdir=\"${dataset}/${task}/${model}\"
            )
            
            
            python src/experiments/main.py --multirun "${HYDRA_ARGS[@]}"
        done
    done
done

for dataset in pems-bay metr-la; do
    for task in short; do
        for model in agcrn astgcn dstagnn stgcn gwnet sttn staeformer autoformer patchtst dlinear lstm itransformer fedformer; do
            HYDRA_ARGS=(
                dataset=${dataset}
                task=${task}
                model=${model} 
                dataset.train_point=0.63
                exp.max_epochs=1
                hydra.sweep.dir=few-shot-baseline 
                hydra.sweep.subdir=\"${dataset}/${task}/${model}\"
            )
            
            python src/experiments/main.py --multirun "${HYDRA_ARGS[@]}"
        done
    done
done
