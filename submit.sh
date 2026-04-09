#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err

set -euo pipefail

mkdir -p logs

# ── Rendezvous info derived from SLURM ──────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE   # GPUs per node

echo "Master: $MASTER_ADDR:$MASTER_PORT  |  Nodes: $NNODES  |  GPUs/node: $NPROC_PER_NODE"

source ../Hackhaton-PyTorch/.venv/bin/activate

# GPU monitoring en background
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 30
done > logs/gpu_monitor.log 2>&1 &
GPU_MON_PID=$!

# ── Launch one torchrun per node via srun ────────────────────────────────────
srun python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    train.py \
        --data_dir      /home/data/ \
        --checkpoint_path checkpoint.pt \
        --seq_len       1024 \
        --n_layer       24 \
        --n_head        16 \
        --n_embd        2048 \
        --batch_size    64 \
        --grad_accum_steps 2 \
        --max_lr        8e-4 \
        --min_lr        8e-5 \
        --warmup_steps  50 \
        --max_steps        1000 \
        --time_limit_min   10

kill $GPU_MON_PID 2>/dev/null