#!/bin/bash
#SBATCH -o /dev/null
#SBATCH --partition=i64m1tga800u
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=low

source scripts/exp/hpc/init.sh

exec "$@"