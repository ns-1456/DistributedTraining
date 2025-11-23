# CS336 Assignment 2 Checklist (Project 3)

This document maps every **problem** and **deliverable** from CS336 Assignment 2 PDF (in workspace root: `cs336_spring2025_assignment2_systems.pdf`) to this repo. We use **CUDA** kernels (not Triton) for Flash Attention 2; the assignment uses Triton—functionality is equivalent.

**Status: All components implemented.**

---

## §1.1 Profiling and Benchmarking

| Problem | Points | Our implementation | How to run |
|--------|--------|--------------------|------------|
| **benchmarking_script** | 4 | `cs336_systems/benchmark.py`: Configurable model init from CONFIGS, random batch, warmup + timed steps, `timeit.default_timer()`, `torch.cuda.synchronize()` after each step. Modes: `forward`, `forward_backward`, `full_step`. | `python -m cs336_systems.benchmark --config small --seq_len 512 --mode forward_backward` |
| (b) Timings | — | Same script; `--n_warmup 5`, `--n_steps 10`. | Report mean ± std forward/backward in writeup. |
| (c) No warmup | — | Re-run with `--n_warmup 0` (or 1, 2) and compare. | Compare warmup vs no-warmup; answer in writeup. |
| **nsys_profile** | 5 | NVTX ranges in `benchmark.py` (`use_nvtx=True`). | `nsys profile -o result python -m cs336_systems.benchmark --config small --mode forward --nvtx` |
| **mixed_precision_accumulation** | 1 | — | Run the four code blocks from PDF; comment on FP16 vs FP32 accumulation in writeup. |
| **benchmarking_mixed_precision** | 2 | `--mixed_precision` uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. | `python -m cs336_systems.benchmark --config small --mixed_precision --compare_precision` |
| **memory_profiling** | 4 | `--memory_profile`: `torch.cuda.memory._record_memory_history()`, run step, `_dump_snapshot("memory_snapshot_*.pickle")`. | `python -m cs336_systems.benchmark --config small --memory_profile`. Load snapshot at pytorch.org/memory_viz. |
| (Sweep + JSON) | — | `--configs`, `--seq_lens`, `--modes`, `--compare_precision`, `--output_json results.json`. | `python -m cs336_systems.benchmark --configs small medium --seq_lens 128 256 512 --output_json out.json` |
| (torch.compile) | — | `--compile` wraps model with `torch.compile()`. | `python -m cs336_systems.benchmark --config small --compile` |

---

## §1.2–1.3 Attention and Flash Attention 2

| Problem | Points | Our implementation | How to run |
|--------|--------|--------------------|------------|
| **pytorch_attention** | 2 | `cuda/benchmark_fa2.py`: batch=8, single head; sweep d_head ∈ [16,32,64,128], seq_len ∈ [256,1024,4096,8192,16384]; warmup + sync, fwd/back timings, memory. | `python -m cuda.benchmark_fa2 --mode pytorch_attn --backward` |
| **torch_compile** | 2 | `compiled_attention` (naive attention under `@torch.compile`) in `cuda/benchmark_fa2.py`. FA2 vs naive vs compiled comparison. | `python -m cuda.benchmark_fa2 --mode flash --backward` |
| **flash_forward** (PyTorch tiled) | — | `cs336_systems/flash_attention.py`: Pure PyTorch `FlashAttentionPyTorch` autograd Function; tiled fwd+bwd, online softmax, causal. | Tests: `pytest tests/test_attention.py` |
| **flash_forward** (CUDA) | 15 | `cuda/flash_attention.cu` + `cuda/flash_attn_func.py`: Flash Attention 2 **CUDA** forward (tiled, online softmax, causal), returns O and L. | Build: `cd cuda && pip install -e .` |
| **flash_backward** | 5 | `cuda/flash_attn_func.py`: CUDA backward with recomputation (L, Q, K, V); eqs 13–19. | `python -m cuda.benchmark_fa2 --mode flash --backward` |
| **flash_benchmarking** | 5 | `cuda/benchmark_fa2.py`: sweep seq_len (e.g. 256–16384), d_head (16–128), naive vs cuda_fa2 vs torch.compile; batch 1, causal. | `python -m cuda.benchmark_fa2 --mode flash --seq_lens 128 256 512 1024 --d_heads 16 32 64 --backward --output_json fa2_results.json` |
| Model integration | — | `cs336_systems/model.py`: `MultiHeadAttention(use_flash=True)` swaps attention implementation. | `train.py --use_flash` |

---

## §2 Distributed Data Parallel

| Problem | Points | Our implementation | How to run |
|--------|--------|--------------------|------------|
| **DDP** | — | `cs336_systems/ddp.py`: data utilities, `setup_distributed`, `DistributedSampler`. `cs336_systems/ddp_custom.py`: `DDPIndividualParameters` (per-param async all-reduce), `DDPBucketed` (bucketed gradient communication). | `torchrun --nproc_per_node=2 train.py --ddp --ddp_mode pytorch` (or `individual`, `bucketed`) |
| Gradient accumulation | — | Supported in `train.py`. | — |
| Mixed precision | — | `--mixed_precision` with `GradScaler`. | `train.py --ddp --mixed_precision` |

---

## §2 Optimizer state sharding

| Problem | Points | Our implementation | How to run |
|--------|--------|--------------------|------------|
| **Optimizer sharding (ZeRO-1)** | — | `cs336_systems/optimizer_sharding.py`: `ShardedOptimizer` wraps any optimizer; partitions m,v by parameter shard, broadcast updated params. | `train.py --ddp --sharded_optimizer` |
| **compare_memory** | — | `compare_memory(model)` utility in `optimizer_sharding.py` for memory comparison. | Import and call for memory profiling. |

---

## Tests

| Test suite | Coverage | How to run |
|------------|----------|------------|
| `tests/adapters.py` | Adapter functions for all implementations (FlashAttention PyTorch/CUDA, DDP individual/bucketed, ShardedOptimizer). | Used by other tests. |
| `tests/test_attention.py` | Flash attention forward + backward (PyTorch and CUDA). | `pytest tests/test_attention.py` |
| `tests/test_ddp.py` | Bucketed DDP (`DDPBucketed`). | `pytest tests/test_ddp.py` |
| `tests/test_ddp_individual_parameters.py` | Per-param DDP (`DDPIndividualParameters`). | `pytest tests/test_ddp_individual_parameters.py` |
| `tests/test_sharded_optimizer.py` | ZeRO-1 `ShardedOptimizer`. | `pytest tests/test_sharded_optimizer.py` |

---

## Model configurations (Table 1)

| Config | d_model | d_ff | num_layers | num_heads |
|--------|---------|------|------------|-----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7b | 2560 | 10240 | 32 | 32 |

Defined in `cs336_systems/model.py` (`CONFIGS`).

---

## Summary: quick reference

| Component | Command |
|-----------|---------|
| Model benchmark | `python -m cs336_systems.benchmark --config small --seq_len 512 --mode forward_backward` |
| Sweep + JSON | `python -m cs336_systems.benchmark --configs small medium --seq_lens 128 256 512 --output_json out.json` |
| Nsight profile | `nsys profile -o result python -m cs336_systems.benchmark --config small --nvtx` |
| Memory snapshot | `python -m cs336_systems.benchmark --config small --memory_profile` |
| PyTorch attention | `python -m cuda.benchmark_fa2 --mode pytorch_attn --backward` |
| Flash attention | `python -m cuda.benchmark_fa2 --mode flash --backward` |
| DDP training | `torchrun --nproc_per_node=2 train.py --ddp --ddp_mode bucketed` |
| DDP + ZeRO-1 | `torchrun --nproc_per_node=2 train.py --ddp --sharded_optimizer` |
| Flash in model | `train.py --use_flash` |
| All tests | `pytest tests/` |
