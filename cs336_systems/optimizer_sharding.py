from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn


class ShardedAdamW:
    """ZeRO Stage 1: optimizer state partitioning across ranks.

    Each rank holds optimizer states (m, v) only for its parameter shard,
    reducing per-device memory by ~world_size for optimizer states.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rank = rank
        self.world_size = world_size
        self.step_count = 0

        self.params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self.flat_params = self._flatten(self.params)
        self.total_size = self.flat_params.numel()

        chunk_size = math.ceil(self.total_size / world_size)
        self.shard_start = rank * chunk_size
        self.shard_end = min(self.shard_start + chunk_size, self.total_size)
        self.shard_size = self.shard_end - self.shard_start

        self.m = torch.zeros(self.shard_size, device=self.flat_params.device, dtype=torch.float32)
        self.v = torch.zeros(self.shard_size, device=self.flat_params.device, dtype=torch.float32)

        self._param_offsets: list[tuple[int, int]] = []
        offset = 0
        for p in self.params:
            n = p.numel()
            self._param_offsets.append((offset, offset + n))
            offset += n

    @staticmethod
    def _flatten(params: list[nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.detach().view(-1) for p in params])

    def _sync_flat_to_params(self) -> None:
        for p, (start, end) in zip(self.params, self._param_offsets):
            p.data.copy_(self.flat_params[start:end].view(p.shape))

    def _sync_params_to_flat(self) -> None:
        for p, (start, end) in zip(self.params, self._param_offsets):
            self.flat_params[start:end] = p.detach().view(-1)

    def _sync_grads_to_flat(self) -> torch.Tensor:
        flat_grad = torch.cat([
            p.grad.detach().view(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            for p in self.params
        ])
        return flat_grad

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        self.step_count += 1

        flat_grad = self._sync_grads_to_flat()

        if self.world_size > 1:
            dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG)

        self._sync_params_to_flat()

        grad_shard = flat_grad[self.shard_start : self.shard_end].float()
        param_shard = self.flat_params[self.shard_start : self.shard_end].float()

        self.m.mul_(self.beta1).add_(grad_shard, alpha=1 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad_shard, grad_shard, value=1 - self.beta2)

        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)

        update = m_hat / (v_hat.sqrt() + self.eps)

        if self.weight_decay > 0:
            update = update + self.weight_decay * param_shard

        param_shard = param_shard - self.lr * update
        self.flat_params[self.shard_start : self.shard_end] = param_shard.to(
            self.flat_params.dtype
        )

        if self.world_size > 1:
            dist.all_gather_into_tensor(
                self.flat_params,
                self.flat_params[self.shard_start : self.shard_end],
            )

        self._sync_flat_to_params()

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "m": self.m,
            "v": self.v,
            "lr": self.lr,
            "rank": self.rank,
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]
        self.m.copy_(state["m"])
        self.v.copy_(state["v"])
        self.lr = state["lr"]


def compare_memory(model: nn.Module, device: str = "cuda") -> None:
    """Print memory comparison between standard AdamW and ShardedAdamW."""
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 4

    standard_opt_mem = total_params * bytes_per_param * 2
    standard_total = total_params * bytes_per_param + standard_opt_mem

    for ws in [1, 2, 4, 8]:
        sharded_opt_mem = (total_params * bytes_per_param * 2) / ws
        sharded_total = total_params * bytes_per_param + sharded_opt_mem

        savings = (1 - sharded_total / standard_total) * 100
        print(
            f"World size {ws}: "
            f"Standard={standard_total / 1e6:.1f}MB, "
            f"Sharded={sharded_total / 1e6:.1f}MB, "
            f"Savings={savings:.1f}%"
        )
