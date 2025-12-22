from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class ShardedOptimizer:
    """ZeRO-1: optimizer state partitioning across ranks.

    Wraps an arbitrary optimizer class so that each rank only maintains
    optimizer states for its shard of parameters.  After each step the
    updated parameter data is broadcast to all ranks.
    """

    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Deduplicate parameters (handles tied weights) while preserving order.
        seen: set[int] = set()
        all_params: list[nn.Parameter] = []
        for p in params:
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                all_params.append(p)
        self.all_params = all_params

        # Round-robin ownership: rank i owns param at index j when j % world_size == i.
        self.param_to_rank = {
            j: j % self.world_size for j in range(len(self.all_params))
        }
        owned_params = [
            p
            for j, p in enumerate(self.all_params)
            if self.param_to_rank[j] == self.rank
        ]

        # Each rank creates an optimizer only for the parameters it owns.
        self.local_optimizer = optimizer_cls(owned_params, **kwargs)

    def zero_grad(self):
        for p in self.all_params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.local_optimizer.step()
        for j, p in enumerate(self.all_params):
            dist.broadcast(p.data, src=self.param_to_rank[j])


def get_sharded_optimizer(params, optimizer_cls, **kwargs):
    """Return a ShardedOptimizer wrapping *optimizer_cls* on *params*."""
    return ShardedOptimizer(params, optimizer_cls, **kwargs)


def compare_memory(model: nn.Module, device: str = "cuda") -> None:
    """Print memory comparison between standard AdamW and ShardedOptimizer."""
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
