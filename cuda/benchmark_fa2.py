from __future__ import annotations

import argparse
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
    n_warmup: int = 3,
    n_steps: int = 10,
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
            torch.cuda.synchronize()
            t2 = timeit.default_timer()
            out.sum().backward()
            torch.cuda.synchronize()
            t3 = timeit.default_timer()
            bwd_times.append(t3 - t2)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results: dict[str, float] = {
        "fwd_mean_ms": sum(fwd_times) / len(fwd_times) * 1000,
        "fwd_std_ms": (
            sum((t - sum(fwd_times) / len(fwd_times)) ** 2 for t in fwd_times) / len(fwd_times)
        ) ** 0.5 * 1000,
        "peak_mem_mb": peak_mem,
    }
    if backward and bwd_times:
        results["bwd_mean_ms"] = sum(bwd_times) / len(bwd_times) * 1000

    return results


def run_sweep(
    seq_lens: list[int],
    d_heads: list[int],
    batch_size: int = 8,
    num_heads: int = 8,
    is_causal: bool = True,
    backward: bool = False,
) -> None:
    header = f"{'Method':<18} {'seq_len':>8} {'d_head':>6} {'fwd_ms':>10} {'bwd_ms':>10} {'mem_MB':>10}"
    print(header)
    print("-" * len(header))

    methods: dict[str, object] = {"naive": naive_attention}

    try:
        _ = compiled_attention(
            torch.randn(1, 1, 4, 32, device="cuda"),
            torch.randn(1, 1, 4, 32, device="cuda"),
            torch.randn(1, 1, 4, 32, device="cuda"),
            True,
        )
        methods["torch.compile"] = compiled_attention
    except Exception:
        pass

    if HAS_CUDA_FA2:
        methods["cuda_fa2"] = flash_attention_2

    for seq_len in seq_lens:
        for d_head in d_heads:
            bs = batch_size if seq_len <= 4096 else max(1, batch_size // 4)

            for name, fn in methods.items():
                try:
                    Q = torch.randn(bs, num_heads, seq_len, d_head, device="cuda")
                    K = torch.randn(bs, num_heads, seq_len, d_head, device="cuda")
                    V = torch.randn(bs, num_heads, seq_len, d_head, device="cuda")

                    res = benchmark_attention(fn, Q, K, V, is_causal, backward=backward)
                    bwd_str = f"{res.get('bwd_mean_ms', 0):.2f}" if backward else "n/a"
                    print(
                        f"{name:<18} {seq_len:>8} {d_head:>6} "
                        f"{res['fwd_mean_ms']:>10.2f} {bwd_str:>10} "
                        f"{res['peak_mem_mb']:>10.1f}"
                    )
                except RuntimeError as e:
                    print(f"{name:<18} {seq_len:>8} {d_head:>6} {'OOM' if 'out of memory' in str(e) else 'ERR':>10}")
                    torch.cuda.empty_cache()

            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention 2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--seq_lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--d_heads", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    run_sweep(
        seq_lens=args.seq_lens,
        d_heads=args.d_heads,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        backward=args.backward,
    )


if __name__ == "__main__":
    main()
