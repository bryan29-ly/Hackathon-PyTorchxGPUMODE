#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err

set -euo pipefail

mkdir -p logs

source ../Hackhaton-PyTorch/.venv/bin/activate

# Optional first arg overrides checkpoint path.
CHECKPOINT_PATH="${1:-checkpoint.pt}"

echo "Running eval with checkpoint: ${CHECKPOINT_PATH}"

srun python eval.py \
    --data_dir /home/data/ \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --batch_size 8 \
    --val_fraction 0.01

