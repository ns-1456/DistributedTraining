from __future__ import annotations

import argparse
import json
import math
import timeit

import torch

try:
    from cuda.flash_attn_func import flash_attention_2
    HAS_CUDA_FA2 = True
except ImportError:
    HAS_CUDA_FA2 = False


def naive_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if is_causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


@torch.compile
def compiled_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if is_causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def benchmark_attention(
    fn,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool,
    n_warmup: int = 5,
    n_steps: int = 100,
    backward: bool = False,
) -> dict[str, float]:
    for _ in range(n_warmup):
        Q_w = Q.clone().requires_grad_(backward)
        K_w = K.clone().requires_grad_(backward)
        V_w = V.clone().requires_grad_(backward)
        out = fn(Q_w, K_w, V_w, is_causal)
        if backward:
            out.sum().backward()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    fwd_times: list[float] = []
    bwd_times: list[float] = []

    for _ in range(n_steps):
        Q_r = Q.clone().requires_grad_(backward)
        K_r = K.clone().requires_grad_(backward)
        V_r = V.clone().requires_grad_(backward)

        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        out = fn(Q_r, K_r, V_r, is_causal)
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        fwd_times.append(t1 - t0)

        if backward:
            mem_before_bwd = torch.cuda.memory_allocated() / (1024 * 1024)
            torch.cuda.synchronize()
            t2 = timeit.default_timer()
            out.sum().backward()
            torch.cuda.synchronize()
            t3 = timeit.default_timer()
            bwd_times.append(t3 - t2)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fwd_mean = sum(fwd_times) / len(fwd_times)
    fwd_var = sum((t - fwd_mean) ** 2 for t in fwd_times) / len(fwd_times)
    results: dict[str, float] = {
        "fwd_mean_ms": fwd_mean * 1000,
        "fwd_std_ms": fwd_var ** 0.5 * 1000,
        "peak_mem_mb": peak_mem,
    }
    if backward and bwd_times:
        bwd_mean = sum(bwd_times) / len(bwd_times)
        results["bwd_mean_ms"] = bwd_mean * 1000

    return results


def run_pytorch_attention_benchmark(batch_size: int = 8) -> list[dict]:
    """Assignment §1.2: benchmark naive attention across d_head and seq_len combinations."""
    d_heads = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    all_results: list[dict] = []

    header = f"{'d_head':>6} {'seq_len':>8} {'fwd_ms':>10} {'bwd_ms':>10} {'mem_MB':>10}"
    print("=== PyTorch Naive Attention Benchmark (batch=8, single head) ===")
    print(header)
    print("-" * len(header))

    for d_head in d_heads:
        for seq_len in seq_lens:
            try:
                Q = torch.randn(batch_size, seq_len, d_head, device="cuda")
                K = torch.randn(batch_size, seq_len, d_head, device="cuda")
                V = torch.randn(batch_size, seq_len, d_head, device="cuda")

                torch.cuda.reset_peak_memory_stats()
                res = benchmark_attention(
                    naive_attention, Q, K, V, is_causal=False, backward=True
                )
                entry = {"d_head": d_head, "seq_len": seq_len, "method": "naive", **res}
                all_results.append(entry)
                print(
                    f"{d_head:>6} {seq_len:>8} "
                    f"{res['fwd_mean_ms']:>10.2f} "
                    f"{res.get('bwd_mean_ms', 0):>10.2f} "
                    f"{res['peak_mem_mb']:>10.1f}"
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{d_head:>6} {seq_len:>8} {'OOM':>10}")
                    torch.cuda.empty_cache()
                    all_results.append({"d_head": d_head, "seq_len": seq_len, "method": "naive", "oom": True})
                else:
                    raise

    return all_results


def run_flash_benchmark(
    seq_lens: list[int],
    d_heads: list[int],
    batch_size: int = 1,
    is_causal: bool = True,
    backward: bool = False,
) -> list[dict]:
    """Assignment flash_benchmarking: compare naive vs compiled vs FA2."""
    methods: dict[str, object] = {"naive": naive_attention}

    try:
        warm = torch.randn(1, 4, 32, device="cuda")
        compiled_attention(warm, warm, warm, True)
        methods["torch.compile"] = compiled_attention
    except Exception:
        pass

    if HAS_CUDA_FA2:
        methods["cuda_fa2"] = flash_attention_2

    header = f"{'Method':<18} {'seq_len':>8} {'d_head':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'mem_MB':>10}"
    print("\n=== Flash Attention Benchmark ===")
    print(header)
    print("-" * len(header))

    all_results: list[dict] = []

    for seq_len in seq_lens:
        for d_head in d_heads:
            for name, fn in methods.items():
                try:
                    Q = torch.randn(batch_size, seq_len, d_head, device="cuda")
                    K = torch.randn(batch_size, seq_len, d_head, device="cuda")
                    V = torch.randn(batch_size, seq_len, d_head, device="cuda")

                    torch.cuda.reset_peak_memory_stats()
                    res = benchmark_attention(fn, Q, K, V, is_causal, backward=backward)
                    bwd_str = f"{res.get('bwd_mean_ms', 0):.2f}" if backward else "n/a"
                    entry = {
                        "method": name, "seq_len": seq_len, "d_head": d_head,
                        "is_causal": is_causal, **res,
                    }
                    all_results.append(entry)
                    print(
                        f"{name:<18} {seq_len:>8} {d_head:>6} "
                        f"{res['fwd_mean_ms']:>10.2f} {bwd_str:>10} "
                        f"{res['peak_mem_mb']:>10.1f}"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{name:<18} {seq_len:>8} {d_head:>6} {'OOM':>10}")
                        torch.cuda.empty_cache()
                    else:
                        raise
            print()

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Attention Implementations")
    parser.add_argument("--mode", default="flash", choices=["pytorch_attn", "flash", "all"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--seq_lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--d_heads", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    all_results: list[dict] = []

    if args.mode in ("pytorch_attn", "all"):
        res = run_pytorch_attention_benchmark(batch_size=args.batch_size)
        all_results.extend(res)

    if args.mode in ("flash", "all"):
        res = run_flash_benchmark(
            seq_lens=args.seq_lens,
            d_heads=args.d_heads,
            batch_size=1,
            backward=args.backward,
        )
        all_results.extend(res)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
