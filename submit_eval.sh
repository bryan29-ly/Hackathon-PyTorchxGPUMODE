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

VENV_ACTIVATE="${VENV_ACTIVATE:-../Hackhaton-PyTorch/.venv/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
    source "${VENV_ACTIVATE}"
else
    echo "[warn] venv activate script not found at ${VENV_ACTIVATE}; continuing with system python"
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "[error] neither python nor python3 is available in PATH"
    exit 127
fi

# Optional first arg overrides checkpoint path.
CHECKPOINT_PATH="${1:-checkpoint.pt}"
MAX_BATCHES="${MAX_BATCHES:-2000}"
LOG_EVERY="${LOG_EVERY:-100}"

echo "Running eval with checkpoint: ${CHECKPOINT_PATH}"
echo "Using python executable: ${PYTHON_BIN}"
echo "Eval controls: max_batches=${MAX_BATCHES}, log_every=${LOG_EVERY}"

srun "${PYTHON_BIN}" eval.py \
    --data_dir /home/data/ \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --batch_size 8 \
    --val_fraction 0.01 \
    --max_batches "${MAX_BATCHES}" \
    --log_every "${LOG_EVERY}"

