# CS336 Assignment 2 Checklist (Project 3)

This document maps every **problem** and **deliverable** from CS336 Assignment 2 PDF (in workspace root: `cs336_spring2025_assignment2_systems.pdf`) to this repo. We use **CUDA** kernels (not Triton) for Flash Attention 2; the assignment uses Triton—functionality is equivalent.

---

## §1.1 Profiling and Benchmarking

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **benchmarking_script** | 4 | `cs336_systems/benchmark.py`: `benchmark_model(config_name, batch_size, seq_len, n_warmup, n_steps, mode, mixed_precision, device)`. Initializes model from CONFIGS (Table 1 sizes), random batch, warmup then timed steps, `timeit.default_timer()`, `torch.cuda.synchronize()` after each step. Modes: forward, forward_backward, full_step. | **Notebook §1–§2**: Run benchmark for small/medium, seq 128/256/512; table and plots. |
| (b) Timings | — | Same script; use 5 warmup, 10 steps. | Report mean ± std forward/backward in writeup. |
| (c) No warmup | — | Re-run with 0, 1, 2 warmup; compare. | **Notebook**: Add cell for warmup vs no-warmup comparison; answer in writeup. |
| **nsys_profile** | 5 | — | Run `nsys profile -o result python -m cs336_systems.benchmark ...`; use NVTX ranges in model if desired. Answer (a)–(e) in writeup from profile. |
| **mixed_precision_accumulation** | 1 | — | Run the four code blocks from PDF; comment on FP16 vs FP32 accumulation in writeup. |
| **benchmarking_mixed_precision** | 2 | `benchmark_model(..., mixed_precision=True)` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. | **Notebook §3**: FP32 vs BF16 comparison table and bar chart. (a)(b) ToyModel dtypes and LayerNorm in writeup.) |
| **memory_profiling** | 4 | Add option to benchmark script: `torch.cuda.memory._record_memory_history`, run step, `_dump_snapshot("memory_snapshot.pickle")`. | Add notebook cell or script path; load snapshot in pytorch.org/memory_viz. Deliver (a) two timelines, (b) peak table, (c) mixed-precision peak, (d) residual stream size, (e) largest allocations in writeup. |

---

## §1.2–1.3 Attention and Flash Attention 2

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **pytorch_attention** | 2 | Standalone attention benchmark: batch=8, single head; sweep d in [16,32,64,128], seq in [256,1024,4096,8192,16384]; time 100 fwd, memory before backward, 100 backward; warmup + sync. | Implement in `cuda/benchmark_fa2.py` or separate script; report table, OOM size, memory accounting, paragraph in writeup. |
| **torch_compile** | 2 | — | (a) Add compiled attention to attention benchmark script; table vs uncompiled. (b) `torch.compile(model)` in benchmark script; table vanilla vs compiled. |
| **flash_forward** (PyTorch tiled) | — | Optional: PyTorch autograd Function with tiled FA2 forward (no Triton). | For debugging; we have CUDA kernel. |
| **flash_forward** (Triton/CUDA) | 15 | `cuda/flash_attention.cu`: Flash Attention 2 **CUDA** forward (tiled, online softmax, causal), returns O and L. `cuda/flash_attn_func.py`: `FlashAttention2` autograd. | Build: `cd cuda && pip install -e .`; **Notebook** or `benchmark_fa2.py`: compare FA2 vs vanilla. |
| **flash_backward** | 5 | Backward with recomputation (L, Q, K, V); D = rowsum(O◦dO); eqs 13–19. Implemented in CUDA or torch.compile in our repo. | `benchmark_fa2.py`: forward+backward timings. |
| **flash_benchmarking** | 5 | `cuda/benchmark_fa2.py`: sweep seq_len (e.g. 128–65536), d_head (16–128), BF16/FP32; batch 1, causal; report table. | Run script; put table in writeup. |

---

## §2 Distributed Data Parallel

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **DDP** | — | `cs336_systems/ddp.py`: `DistributedDataParallel`, `DistributedSampler`, init with `torchrun` or `mp.spawn`, gradient accumulation, mixed precision. | `torchrun --nproc_per_node=2 cs336_systems/ddp.py ...` (or run `train.py --ddp`). Document in README; Kaggle dual-GPU instructions. |

---

## §2 Optimizer state sharding

| Problem | Points | Our implementation | Notebook / How to run |
|--------|--------|--------------------|------------------------|
| **Optimizer sharding (ZeRO-1)** | — | `cs336_systems/optimizer_sharding.py`: `ShardedAdamW` — partition m,v by parameter shard, all-reduce grads, local update, all-gather params. | Use with DDP; optional notebook cell for memory comparison. |

---

## Model sizing (Table 1)

Our `CONFIGS` in `cs336_systems/model.py`: small (768, 3072, 12, 12), medium (1024, 4096, 24, 16), large (1280, 5120, 36, 20), xl (1600, 6400, 48, 25). We do not have 2.7B in the table; you can add it (2560, 10240, 32, 32) for memory profiling if desired.

---

## Summary: what the notebook runs

- **§1** Benchmark forward/forward_backward for small & medium, multiple seq lengths; table.
- **§2** Plots: forward time and fwd+bwd time vs seq_len.
- **§3** Mixed precision: FP32 vs BF16 comparison.

## What to add for full assignment

- **Notebook**: (1) Warmup vs no-warmup (and 1–2 warmup) timing cell. (2) Optional memory profiling cell (record history, dump snapshot, note path to memory_viz). (3) Optional attention-only benchmark (sweep d_head and seq_len) and torch.compile comparison.
- **Writeup**: Nsight (a)–(e), mixed_precision_accumulation and ToyModel dtypes, memory_profiling (a)–(e), pytorch_attention table and paragraph, torch_compile tables, flash_benchmarking table.
