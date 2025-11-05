from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cs336_systems.model import CONFIGS, TransformerLM


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


class RandomTokenDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int = 10000) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


class TextFileDataset(Dataset):
    def __init__(self, path: str, seq_len: int, vocab_size: int = 10000) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        if os.path.exists(path):
            raw = torch.load(path, weights_only=True)
            if isinstance(raw, torch.Tensor):
                self.data = raw
            else:
                self.data = torch.randint(0, vocab_size, (1000, seq_len + 1))
        else:
            self.data = torch.randint(0, vocab_size, (1000, seq_len + 1))
        if self.data.dim() == 1:
            n_chunks = len(self.data) // (seq_len + 1)
            self.data = self.data[: n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


def train_ddp(
    rank: int,
    world_size: int,
    config: str,
    train_data_path: str | None,
    epochs: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    save_path: str | None,
    mixed_precision: bool = False,
    accumulate_steps: int = 1,
) -> None:
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    cfg = CONFIGS[config]
    model = TransformerLM(
        vocab_size=10000,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=seq_len,
    ).to(device)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    if train_data_path:
        dataset = TextFileDataset(train_data_path, seq_len)
    else:
        dataset = RandomTokenDataset(num_samples=2048, seq_len=seq_len)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                loss = loss / accumulate_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulate_steps
            n_batches += 1

        if rank == 0:
            avg_loss = total_loss / max(n_batches, 1)
            tokens_per_batch = batch_size * seq_len * world_size
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {min(avg_loss, 20):.2f} | "
                f"Tokens/batch: {tokens_per_batch}"
            )

    if rank == 0 and save_path:
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, save_path)
        print(f"Model saved to {save_path}")

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--config", default="small", choices=list(CONFIGS.keys()))
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--use_spawn", action="store_true", help="Use mp.spawn instead of torchrun")
    args = parser.parse_args()

    if args.use_spawn:
        mp.spawn(
            train_ddp,
            args=(
                args.world_size,
                args.config,
                args.data_path,
                args.epochs,
                args.batch_size,
                args.seq_len,
                args.lr,
                args.save_path,
                args.mixed_precision,
                args.accumulate_steps,
            ),
            nprocs=args.world_size,
            join=True,
        )
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        train_ddp(
            rank=rank,
            world_size=world_size,
            config=args.config,
            train_data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            save_path=args.save_path,
            mixed_precision=args.mixed_precision,
            accumulate_steps=args.accumulate_steps,
        )


if __name__ == "__main__":
    main()
