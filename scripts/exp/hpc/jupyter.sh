#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=debug
#SBATCH --output=jupyter.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=high

source /hpc2hdd/home/yfeng083/miniconda3/etc/profile.d/conda.sh
source activate /hpc2hdd/home/yfeng083/miniconda3/envs/astrology

jupyter notebook --no-browser --port=8889 --ip=$(hostname)