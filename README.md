# ML Systems: Flash Attention 2, DDP, and Optimizer Sharding

A CS336 Assignment 2–style systems project implementing GPU-optimized Transformer training infrastructure from scratch: a custom Flash Attention 2 CUDA kernel, distributed data parallel (DDP) training, and ZeRO-1 optimizer state sharding. Includes a benchmarking harness for profiling model configurations under mixed precision.

## Features

- **Transformer LM** — from-scratch PyTorch Transformer with RMSNorm, RoPE, SwiGLU, manual `nn.Parameter` matmuls (no `nn.Linear`)
- **Benchmarking harness** — time forward/backward/full-step across model sizes (small → XL), mixed precision (bf16)
- **Flash Attention 2 CUDA kernel** — tiled forward and backward passes with shared memory, causal masking, online softmax
- **Distributed Data Parallel** — multi-GPU training via `torchrun` or `mp.spawn`, gradient accumulation, mixed precision
- **ZeRO-1 optimizer sharding** — `ShardedAdamW` partitions optimizer states across ranks, all-reduce gradients, all-gather parameters
- **Mixed precision** — `torch.autocast` with bf16 + `GradScaler` throughout

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Demo notebook (benchmark + plots)

From the repo root `project-3-systems/`, open `notebooks/demo_benchmark.ipynb`. Run all cells to benchmark forward/backward timings by config and sequence length, and compare FP32 vs BF16 (requires CUDA).

### Benchmark model

```bash
python -m cs336_systems.benchmark --config small --batch_size 4 --seq_len 512 --mode forward_backward --mixed_precision
```

### Build CUDA extension

```bash
cd cuda
pip install -e .
cd ..
```

### Run Flash Attention 2 benchmark

```bash
python -m cuda.benchmark_fa2 --backward --seq_lens 256 512 1024 2048 4096
```

### Train (single GPU)

```bash
python train.py --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

### Train with DDP (2 GPUs)

```bash
torchrun --nproc_per_node=2 train.py --ddp --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

### Train with DDP + ZeRO-1

```bash
torchrun --nproc_per_node=2 train.py --ddp --sharded_optimizer --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

## Kaggle Dual-GPU Setup (T4 x 2)

1. Create a new Kaggle notebook with **GPU T4 x 2** accelerator
2. Upload this project as a dataset or clone from GitHub
3. Install dependencies:
   ```python
   !pip install -r /kaggle/input/project-3-systems/requirements.txt
   ```
4. Build CUDA extension:
   ```python
   %cd /kaggle/input/project-3-systems/cuda
   !pip install -e .
   %cd /kaggle/input/project-3-systems
   ```
5. Launch DDP training:
   ```python
   !torchrun --nproc_per_node=2 train.py --ddp --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
   ```
6. For sharded optimizer:
   ```python
   !torchrun --nproc_per_node=2 train.py --ddp --sharded_optimizer --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
   ```

## Project Structure

```
project-3-systems/
├── cs336_systems/
│   ├── __init__.py
│   ├── model.py              # Transformer LM (RMSNorm, RoPE, MHA, SwiGLU)
│   ├── benchmark.py           # Benchmarking harness
│   ├── ddp.py                 # Distributed data parallel training
│   └── optimizer_sharding.py  # ZeRO-1 ShardedAdamW
├── cuda/
│   ├── __init__.py
│   ├── flash_attention.cu     # Flash Attention 2 CUDA kernel (fwd + bwd)
│   ├── flash_attn_func.py     # torch.autograd.Function wrapper
│   ├── setup.py               # CUDAExtension build script
│   └── benchmark_fa2.py       # FA2 vs naive vs torch.compile benchmark
├── train.py                   # Top-level training entrypoint
├── requirements.txt
└── README.md
```

## Model Configurations

| Config | d_model | d_ff  | Layers | Heads | ~Params |
|--------|---------|-------|--------|-------|---------|
| small  | 768     | 3072  | 12     | 12    | 85M     |
| medium | 1024    | 4096  | 24     | 16    | 303M    |
| large  | 1280    | 5120  | 36     | 20    | 700M    |
| xl     | 1600    | 6400  | 48     | 25    | 1.4B    |

## Results

### Benchmark Timings (placeholder)

| Config | Mode             | Precision | Mean (ms) | Tokens/s  | Peak Mem (MB) |
|--------|------------------|-----------|-----------|-----------|---------------|
| small  | forward          | fp32      | —         | —         | —             |
| small  | forward          | bf16      | —         | —         | —             |
| small  | forward_backward | fp32      | —         | —         | —             |
| small  | forward_backward | bf16      | —         | —         | —             |
| medium | forward_backward | bf16      | —         | —         | —             |

### Flash Attention 2 vs Vanilla (placeholder)

| seq_len | d_head | Naive (ms) | torch.compile (ms) | CUDA FA2 (ms) | Speedup |
|---------|--------|------------|---------------------|---------------|---------|
| 1024    | 64     | —          | —                   | —             | —       |
| 4096    | 64     | —          | —                   | —             | —       |
| 8192    | 64     | —          | —                   | —             | —       |

### DDP Scaling (placeholder)

| GPUs | Config | Throughput (tok/s) | Loss  |
|------|--------|--------------------|-------|
| 1    | small  | —                  | —     |
| 2    | small  | —                  | —     |

## References

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — Stanford, Spring 2025
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2020
