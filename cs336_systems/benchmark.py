from __future__ import annotations

import argparse
import timeit

import torch

from cs336_systems.model import CONFIGS, TransformerLM


def benchmark_model(
    config_name: str = "small",
    batch_size: int = 4,
    seq_len: int = 512,
    n_warmup: int = 3,
    n_steps: int = 10,
    mode: str = "forward",
    mixed_precision: bool = False,
    device: str = "cuda",
) -> dict[str, float]:
    cfg = CONFIGS[config_name]
    model = TransformerLM(
        vocab_size=10000,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=seq_len,
    ).to(device)

    optimizer = None
    if mode == "full_step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    def _step() -> None:
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            logits = model(input_ids)
            if mode in ("forward_backward", "full_step"):
                loss = logits.sum()
        if mode in ("forward_backward", "full_step"):
            loss.backward()
        if optimizer is not None:
            optimizer.step()
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        _step()

    timings: list[float] = []
    for _ in range(n_steps):
        start = timeit.default_timer()
        _step()
        timings.append(timeit.default_timer() - start)

    t = torch.tensor(timings)
    total_tokens = batch_size * seq_len
    results = {
        "mean_ms": t.mean().item() * 1000,
        "std_ms": t.std().item() * 1000,
        "tokens_per_sec": total_tokens / t.mean().item(),
    }

    if device == "cuda":
        results["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


def print_results(config_name: str, mode: str, mixed_precision: bool, results: dict[str, float]) -> None:
    precision_str = "bf16" if mixed_precision else "fp32"
    print(f"\n{'=' * 60}")
    print(f" Config: {config_name} | Mode: {mode} | Precision: {precision_str}")
    print(f"{'=' * 60}")
    print(f" {'Metric':<25} {'Value':>15}")
    print(f" {'-' * 40}")
    for key, val in results.items():
        print(f" {key:<25} {val:>15.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Transformer LM")
    parser.add_argument("--config", default="small", choices=list(CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mode", default="forward", choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, benchmarks will be slow and inaccurate.")

    results = benchmark_model(
        config_name=args.config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_warmup=args.n_warmup,
        n_steps=args.n_steps,
        mode=args.mode,
        mixed_precision=args.mixed_precision,
        device=device,
    )
    print_results(args.config, args.mode, args.mixed_precision, results)


if __name__ == "__main__":
    main()
