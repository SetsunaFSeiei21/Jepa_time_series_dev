module purge # 清空旧模块
module load cuda/12.4

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
source activate ~/miniconda3/envs/workspace