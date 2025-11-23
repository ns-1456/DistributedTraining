from __future__ import annotations

import argparse
import json
import timeit
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

from cs336_systems.model import CONFIGS, TransformerLM


def benchmark_model(
    config_name: str = "small",
    batch_size: int = 4,
    seq_len: int = 512,
    n_warmup: int = 5,
    n_steps: int = 10,
    mode: str = "forward",
    mixed_precision: bool = False,
    device: str = "cuda",
    memory_profile: bool = False,
    use_nvtx: bool = False,
    compile_model: bool = False,
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

    if compile_model:
        model = torch.compile(model)

    optimizer = None
    if mode == "full_step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device == "cuda")
    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if mixed_precision and device == "cuda"
        else nullcontext()
    )

    def _step() -> None:
        if optimizer is not None:
            optimizer.zero_grad()
        with amp_ctx:
            logits = model(input_ids)
            if mode in ("forward_backward", "full_step"):
                loss = logits.sum()
        if mode in ("forward_backward", "full_step"):
            scaler.scale(loss).backward()
        if optimizer is not None:
            scaler.step(optimizer)
            scaler.update()
        if device == "cuda":
            torch.cuda.synchronize()

    if use_nvtx:
        nvtx.range_push("warmup")
    for _ in range(n_warmup):
        _step()
    if use_nvtx:
        nvtx.range_pop()

    if memory_profile and device == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    timings: list[float] = []
    if use_nvtx:
        nvtx.range_push("benchmark")
    for _ in range(n_steps):
        start = timeit.default_timer()
        _step()
        timings.append(timeit.default_timer() - start)
    if use_nvtx:
        nvtx.range_pop()

    if memory_profile and device == "cuda":
        snapshot_path = f"memory_snapshot_{config_name}_{mode}_{seq_len}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    t = torch.tensor(timings)
    total_tokens = batch_size * seq_len
    results: dict[str, float] = {
        "mean_ms": t.mean().item() * 1000,
        "std_ms": t.std().item() * 1000,
        "tokens_per_sec": total_tokens / t.mean().item(),
    }

    if device == "cuda":
        results["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


def print_results(
    config_name: str, mode: str, mixed_precision: bool, results: dict[str, float]
) -> None:
    precision_str = "bf16" if mixed_precision else "fp32"
    print(f"\n{'=' * 60}")
    print(f" Config: {config_name} | Mode: {mode} | Precision: {precision_str}")
    print(f"{'=' * 60}")
    print(f" {'Metric':<25} {'Value':>15}")
    print(f" {'-' * 40}")
    for key, val in results.items():
        print(f" {key:<25} {val:>15.2f}")
    print()


def run_sweep(args: argparse.Namespace) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, benchmarks will be slow and inaccurate.")

    configs = args.configs if args.configs else list(CONFIGS.keys())
    seq_lens = args.seq_lens if args.seq_lens else [128, 256, 512, 1024]
    modes = args.modes if args.modes else [args.mode]
    precisions = [True, False] if args.compare_precision else [args.mixed_precision]

    all_results: list[dict] = []

    for config_name in configs:
        for seq_len in seq_lens:
            for mode in modes:
                for mp in precisions:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        res = benchmark_model(
                            config_name=config_name,
                            batch_size=args.batch_size,
                            seq_len=seq_len,
                            n_warmup=args.n_warmup,
                            n_steps=args.n_steps,
                            mode=mode,
                            mixed_precision=mp,
                            device=device,
                            memory_profile=args.memory_profile,
                            use_nvtx=args.nvtx,
                            compile_model=args.compile,
                        )
                        entry = {
                            "config": config_name,
                            "seq_len": seq_len,
                            "mode": mode,
                            "precision": "bf16" if mp else "fp32",
                            **res,
                        }
                        all_results.append(entry)
                        print_results(config_name, mode, mp, res)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"OOM: {config_name} seq={seq_len} mode={mode} mp={mp}")
                            torch.cuda.empty_cache()
                        else:
                            raise

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Transformer LM")
    parser.add_argument("--config", default="small", choices=list(CONFIGS.keys()))
    parser.add_argument("--configs", nargs="+", default=None, help="Run sweep over multiple configs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=None, help="Sweep over seq lengths")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mode", default="forward", choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--modes", nargs="+", default=None, help="Sweep over modes")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compare_precision", action="store_true", help="Run both fp32 and bf16")
    parser.add_argument("--memory_profile", action="store_true", help="Dump CUDA memory snapshot")
    parser.add_argument("--nvtx", action="store_true", help="Tag warmup/benchmark phases with NVTX ranges")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the model")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.configs or args.seq_lens or args.modes or args.compare_precision:
        all_results = run_sweep(args)
    else:
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
            memory_profile=args.memory_profile,
            use_nvtx=args.nvtx,
            compile_model=args.compile,
        )
        print_results(args.config, args.mode, args.mixed_precision, results)
        all_results = [{
            "config": args.config,
            "seq_len": args.seq_len,
            "mode": args.mode,
            "precision": "bf16" if args.mixed_precision else "fp32",
            **results,
        }]

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
