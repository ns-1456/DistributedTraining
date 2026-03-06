# ML Systems: Flash Attention 2, DDP, and Optimizer Sharding

A CS336 Assignment 2-style systems project implementing GPU-optimized Transformer training infrastructure from scratch: custom Flash Attention 2 (PyTorch and CUDA), distributed data parallel (DDP) training variants, and ZeRO-1 optimizer state sharding. Includes a benchmarking harness for profiling model configurations under mixed precision and Nsight Systems integration.

## Features

- **TransformerLM** — RMSNorm, RoPE, SwiGLU, causal masking; configs small through 2.7B; `use_flash=True` for FlashAttention2
- **Benchmark harness** — forward / forward+backward / full-step modes; mixed precision (bf16); NVTX ranges, memory profiling, sweep mode, torch.compile, JSON output
- **Flash Attention 2** — Pure PyTorch autograd FA2 (`flash_attention.py`) and CUDA kernel (`cuda/flash_attention.cu`) with tiled forward+backward, online softmax
- **Custom DDP** — Per-parameter async all-reduce (`DDPIndividualParameters`), bucketed gradients (`DDPBucketed`); test suite adapters
- **ZeRO-1** — `ShardedOptimizer` wraps any optimizer, partitions states across ranks, broadcasts updated params
- **Data utilities** — `RandomTokenDataset`, `TextFileDataset`, distributed setup/cleanup
- **Training** — Single GPU, DDP (pytorch/individual/bucketed), DDP+ZeRO-1; mixed precision; torchrun or mp.spawn
- **Tests** — CS336 Assignment 2-style suite: FlashAttention forward+backward, bucketed DDP, per-param DDP, ZeRO-1

## Project Structure

```
project-3-systems/
├── cs336_systems/
│   ├── __init__.py
│   ├── model.py              # TransformerLM (RMSNorm, RoPE, SwiGLU, causal; use_flash)
│   ├── benchmark.py          # Benchmark harness (modes, sweep, NVTX, memory, --compile, --output_json)
│   ├── flash_attention.py    # Pure PyTorch FA2 autograd (tiled, online softmax)
│   ├── ddp.py                # Data utilities (RandomTokenDataset, TextFileDataset, setup/cleanup)
│   ├── ddp_custom.py         # DDPIndividualParameters, DDPBucketed
│   ├── optimizer_sharding.py # ShardedOptimizer (ZeRO-1)
│   └── ...
├── cuda/
│   ├── flash_attention.cu    # CUDA FA2 kernel (forward + backward)
│   ├── flash_attn_func.py    # Autograd wrapper for CUDA FA2
│   ├── benchmark_fa2.py      # pytorch_attn sweep; flash mode (naive vs compile vs CUDA FA2)
│   └── setup.py
├── tests/
│   ├── adapters.py           # Wiring to our implementations
│   ├── common.py             # ToyModel, helpers
│   ├── test_attention.py     # FlashAttention forward+backward
│   ├── test_ddp.py           # Bucketed DDP
│   ├── test_ddp_individual_parameters.py
│   ├── test_sharded_optimizer.py
│   └── fixtures/             # DDP test data
├── docs/
│   └── CS336_ASSIGNMENT2_CHECKLIST.md
├── scripts/
│   └── kaggle_speedrun.py   # Benchmark + DDP commands for 2x T4
├── train.py
├── requirements.txt
└── README.md
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Benchmark model

```bash
python -m cs336_systems.benchmark --config small --batch_size 4 --seq_len 512 --mode forward_backward --mixed_precision
```

Options: `--nvtx` (NVTX ranges for Nsight Systems), `--memory_profile` (dump CUDA snapshots), `--compile` (torch.compile), `--output_json <path>`. Sweep with `--configs small medium`, `--seq_lens 256 512 1024`, `--modes`, `--compare_precision`.

### Build CUDA extension

```bash
cd cuda
pip install -e .
cd ..
```

### Attention benchmark

```bash
# Naive attention sweep: d_head=[16,32,64,128] x seq_len=[256..16384]
python -m cuda.benchmark_fa2 --mode pytorch_attn

# Compare naive vs torch.compile vs CUDA FA2
python -m cuda.benchmark_fa2 --mode flash --backward --seq_lens 256 512 1024 2048 4096 8192 16384
```

Use `--output_json <path>` to save results.

### Speedrun notebook (TinyStories, dual T4)

Use [notebooks/speedrun_tinystories.ipynb](notebooks/speedrun_tinystories.ipynb) for a minimal **timed** (e.g. 20 min) or **full** pretraining run on TinyStories. Prepares data, runs DDP training, then a short generation demo.

### Train (single GPU)

```bash
python train.py --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

### Train with DDP (2 GPUs)

```bash
torchrun --nproc_per_node=2 train.py --ddp --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

DDP modes: `--ddp_mode pytorch` (default), `--ddp_mode individual`, `--ddp_mode bucketed`. Bucketed: `--bucket_size_mb 25`.

### Train with DDP + ZeRO-1

```bash
torchrun --nproc_per_node=2 train.py --ddp --sharded_optimizer --config small --epochs 2 --batch_size 4 --seq_len 512 --mixed_precision
```

### Run tests

```bash
pytest tests/ -v
```

## Model Configurations

| Config | d_model | d_ff  | Layers | Heads | ~Params |
|--------|---------|-------|--------|-------|---------|
| small  | 768     | 3072  | 12     | 12    | 85M     |
| medium | 1024    | 4096  | 24     | 16    | 303M    |
| large  | 1280    | 5120  | 36     | 20   | 700M    |
| xl     | 1600    | 6400  | 48     | 25     | 1.4B    |
| 2.7b   | 2560    | 10240 | 32     | 32     | 2.7B    |

## Assignment Mapping

Each deliverable maps to the codebase and checklist. See [docs/CS336_ASSIGNMENT2_CHECKLIST.md](docs/CS336_ASSIGNMENT2_CHECKLIST.md) for the full problem-to-implementation mapping (profiling, benchmarking, nsys_profile, mixed precision, memory profiling, pytorch_attention, torch_compile, flash_forward/flash_backward, DDP, ZeRO-1).

.

## References

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — Stanford, Spring 2025
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2020

See notebooks and scripts for Kaggle/Colab.
