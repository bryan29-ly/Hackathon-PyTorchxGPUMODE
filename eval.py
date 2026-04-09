"""
Evaluation script for hackathon checkpoints.

Loads a checkpoint produced by train.py, restores the model via model.get_model,
and computes validation loss on tokenized .bin shards.
"""

import argparse
import glob
import os
from contextlib import nullcontext

import numpy as np
import torch

from model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--data_dir", type=str, default="/home/data")
    parser.add_argument("--token_dtype", type=str, default="uint16")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override sequence length from checkpoint config.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None,
                        help="Device override, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional cap on number of eval batches.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Token stride between windows (defaults to seq_len).")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Fallback fraction of tail shards used for validation.")
    return parser.parse_args()


def load_checkpoint(path: str, map_location: str):
    # Newer torch supports weights_only=True for safer unpickling.
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    return ckpt


def select_val_paths(data_dir: str, val_fraction: float):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not paths:
        raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")

    # Prefer explicit validation naming if present.
    named_val = [p for p in paths if "val" in os.path.basename(p).lower()]
    if named_val:
        return named_val

    # Otherwise reserve a small deterministic tail subset for validation.
    n_val = max(1, int(round(len(paths) * val_fraction)))
    return paths[-n_val:]


def iter_windows(shard: np.memmap, seq_len: int, stride: int):
    if len(shard) <= seq_len:
        return
    last_start = len(shard) - seq_len - 1
    for start in range(0, last_start + 1, stride):
        end = start + seq_len + 1
        chunk = torch.from_numpy(shard[start:end].astype(np.int64))
        if chunk.numel() == seq_len + 1:
            yield chunk[:-1], chunk[1:]


def evaluate(model, val_paths, seq_len: int, batch_size: int, stride: int,
             token_dtype: str, device: str, max_batches: int | None):
    model.eval()
    np_dtype = np.dtype(token_dtype)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()

    total_loss = 0.0
    total_tokens = 0
    seen_batches = 0

    with torch.no_grad():
        batch_x, batch_y = [], []

        for path in val_paths:
            shard = np.memmap(path, dtype=np_dtype, mode="r")
            for x, y in iter_windows(shard, seq_len=seq_len, stride=stride):
                batch_x.append(x)
                batch_y.append(y)

                if len(batch_x) == batch_size:
                    xb = torch.stack(batch_x).to(device)
                    yb = torch.stack(batch_y).to(device)
                    with amp_ctx:
                        _, loss = model(xb, yb)
                    n_tokens = xb.numel()
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    seen_batches += 1
                    batch_x, batch_y = [], []

                    if max_batches is not None and seen_batches >= max_batches:
                        mean_loss = total_loss / max(1, total_tokens)
                        return mean_loss, seen_batches, total_tokens

        if batch_x:
            xb = torch.stack(batch_x).to(device)
            yb = torch.stack(batch_y).to(device)
            with amp_ctx:
                _, loss = model(xb, yb)
            n_tokens = xb.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            seen_batches += 1

    if total_tokens == 0:
        raise RuntimeError("No evaluation windows were generated. Check seq_len/data.")

    mean_loss = total_loss / total_tokens
    return mean_loss, seen_batches, total_tokens


def main():
    args = parse_args()
    
    print(f"[eval] args: {vars(args)}", flush=True)

    device = args.device
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available. Please specify --device cpu or ensure CUDA is set up.")
        device = "cuda"

    ckpt = load_checkpoint(args.checkpoint_path, map_location="cpu")
    if "model" not in ckpt or "config" not in ckpt:
        raise KeyError("Checkpoint must contain 'model' and 'config' keys.")

    config = ckpt["config"]
    seq_len = args.seq_len if args.seq_len is not None else int(config.get("seq_len", 1024))
    stride = args.stride if args.stride is not None else seq_len

    model = get_model(config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    val_paths = select_val_paths(args.data_dir, args.val_fraction)
    print(f"[eval] checkpoint: {args.checkpoint_path}", flush=True)
    print(f"[eval] device: {device}", flush=True)
    print(f"[eval] val shards: {len(val_paths)}", flush=True)
    print(f"[eval] seq_len={seq_len} batch_size={args.batch_size} stride={stride}", flush=True)

    val_loss, num_batches, num_tokens = evaluate(
        model=model,
        val_paths=val_paths,
        seq_len=seq_len,
        batch_size=args.batch_size,
        stride=stride,
        token_dtype=args.token_dtype,
        device=device,
        max_batches=args.max_batches,
    )

    print(f"[eval] batches: {num_batches}", flush=True)
    print(f"[eval] tokens: {num_tokens:,}", flush=True)
    print(f"[eval] val_loss: {val_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
