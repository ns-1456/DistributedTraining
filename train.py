from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from cs336_systems.ddp import RandomTokenDataset, TextFileDataset, setup_distributed, cleanup_distributed
from cs336_systems.model import CONFIGS, TransformerLM
from cs336_systems.optimizer_sharding import ShardedAdamW


def train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    is_distributed = world_size > 1

    if is_distributed:
        setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    cfg = CONFIGS[args.config]
    model = TransformerLM(
        vocab_size=10000,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=args.seq_len,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.sharded_optimizer and is_distributed:
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = ShardedAdamW(raw_model, lr=args.lr, rank=rank, world_size=world_size)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")

    if args.data_path and os.path.exists(args.data_path):
        dataset = TextFileDataset(args.data_path, args.seq_len)
    else:
        dataset = RandomTokenDataset(num_samples=2048, seq_len=args.seq_len)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), drop_last=True)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Config: {args.config} | Params: {total_params / 1e6:.1f}M | Device: {device}")
        print(f"DDP: {is_distributed} | Sharded optimizer: {args.sharded_optimizer}")
        print(f"Mixed precision: {args.mixed_precision}")
        print("-" * 60)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        total_tokens = 0
        epoch_start = time.time()

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if isinstance(optimizer, ShardedAdamW):
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.mixed_precision):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            if isinstance(optimizer, ShardedAdamW):
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_tokens = inputs.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens

        elapsed = time.time() - epoch_start
        avg_loss = total_loss / max(len(loader), 1)
        throughput = total_tokens * world_size / elapsed

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {min(torch.exp(torch.tensor(avg_loss)).item(), 1e6):.1f} | "
                f"Throughput: {throughput:.0f} tok/s | "
                f"Time: {elapsed:.1f}s"
            )

    if rank == 0 and args.save_path:
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), args.save_path)
        print(f"\nModel saved to {args.save_path}")

    if is_distributed:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer LM")
    parser.add_argument("--config", default="small", choices=list(CONFIGS.keys()))
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--sharded_optimizer", action="store_true")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--use_spawn", action="store_true")
    args = parser.parse_args()

    if args.ddp:
        if args.use_spawn:
            mp.spawn(
                train_worker,
                args=(args.world_size, args),
                nprocs=args.world_size,
                join=True,
            )
        else:
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
            train_worker(rank, world_size, args)
    else:
        train_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
