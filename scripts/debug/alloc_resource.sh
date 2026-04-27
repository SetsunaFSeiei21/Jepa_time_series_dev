srun  \
  --job-name=vsdebug_demo \
  --partition=debug \
  --gres=gpu:1 \
  --time=00:30:00 \
  --nodes=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --pty bash 

