from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import Dataset


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
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
