#!/usr/bin/env python3
"""
Kaggle dual T4 speedrun: run minimal benchmarks and print DDP commands for CS336 Assignment 2.

Designed for Kaggle free tier "GPU T4 x 2" (2 x 16GB). Uses T4-safe defaults:
  config=small, batch_size=4, seq_len=256 (or 512), mixed_precision (bf16).

Usage (from project-3-systems/):
  python scripts/kaggle_speedrun.py [--seq_lens 128 256 512] [--no_benchmark] [--output_dir .]

Then run the printed torchrun commands in a separate cell for DDP training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Run from repo root so cs336_systems and cuda are importable (works from repo root or scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    os.chdir(REPO_ROOT)
except OSError:
    pass  # read-only (e.g. Kaggle input) – script may still run if cwd is already correct

import torch

from cs336_systems.benchmark import benchmark_model

# T4-safe: small model, modest batch and seq_len to stay under 16GB per GPU
KAGGLE_DEFAULT_CONFIG = "small"
KAGGLE_DEFAULT_BATCH = 4
KAGGLE_DEFAULT_SEQ_LENS = [128, 256, 512]
KAGGLE_DEFAULT_EPOCHS = 1
KAGGLE_N_WARMUP = 3
KAGGLE_N_STEPS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle dual T4 speedrun")
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=KAGGLE_DEFAULT_SEQ_LENS,
        help="Sequence lengths for benchmark sweep",
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Skip benchmark; only print torchrun commands",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("KAGGLE_SPEEDRUN_OUTPUT", "."),
        help="Directory for benchmark JSON output (on Kaggle use /kaggle/working if repo is read-only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=KAGGLE_DEFAULT_EPOCHS,
        help="Epochs for printed DDP commands",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA. Run this on a Kaggle GPU (T4 x 2) notebook.")
        return

    device = "cuda"
    results = []

    if not args.no_benchmark:
        print("Running T4-safe benchmark (small, batch=4, mixed_precision)...")
        for seq_len in args.seq_lens:
            for mode in ("forward", "forward_backward"):
                for use_mp in (False, True):
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        r = benchmark_model(
                            config_name=KAGGLE_DEFAULT_CONFIG,
                            batch_size=KAGGLE_DEFAULT_BATCH,
                            seq_len=seq_len,
                            n_warmup=KAGGLE_N_WARMUP,
                            n_steps=KAGGLE_N_STEPS,
                            mode=mode,
                            mixed_precision=use_mp,
                            device=device,
                        )
                        prec = "bf16" if use_mp else "fp32"
                        results.append({
                            "config": KAGGLE_DEFAULT_CONFIG,
                            "seq_len": seq_len,
                            "mode": mode,
                            "precision": prec,
                            **r,
                        })
                        print(
                            f"  {mode} seq={seq_len} {prec}: "
                            f"{r['mean_ms']:.1f} ms, peak_mem={r.get('peak_mem_mb', 0):.0f} MB"
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"  OOM: {mode} seq={seq_len}")
                            torch.cuda.empty_cache()
                        else:
                            raise

        out_path = os.path.join(args.output_dir, "kaggle_speedrun_benchmark.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {out_path}")
    else:
        print("Skipping benchmark (--no_benchmark).")

    print("\n" + "=" * 60)
    print("DDP training (run in a new cell with GPU T4 x 2):")
    print("=" * 60)
    base = f"--config {KAGGLE_DEFAULT_CONFIG} --epochs {args.epochs} --batch_size {KAGGLE_DEFAULT_BATCH} --seq_len 512 --mixed_precision"
    print("\n# DDP (PyTorch):")
    print(f"!torchrun --nproc_per_node=2 train.py --ddp {base}")
    print("\n# DDP + ZeRO-1 sharded optimizer:")
    print(f"!torchrun --nproc_per_node=2 train.py --ddp --sharded_optimizer {base}")
    print("\nUse --seq_len 256 if you hit OOM on T4.")


if __name__ == "__main__":
    main()
