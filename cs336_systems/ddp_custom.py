from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(p: torch.Tensor) -> None:
            handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(world_size)


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size()
        self._handles: list[tuple[dist.Work, int]] = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        grad_params = [p for p in self.module.parameters() if p.requires_grad]
        reversed_params = list(reversed(grad_params))

        element_size = reversed_params[0].element_size() if reversed_params else 4
        max_elements = int(bucket_size_mb * 1024 * 1024 / element_size)

        self._buckets: list[list[nn.Parameter]] = []
        current_bucket: list[nn.Parameter] = []
        current_size = 0
        for p in reversed_params:
            current_bucket.append(p)
            current_size += p.numel()
            if current_size >= max_elements:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
        if current_bucket:
            self._buckets.append(current_bucket)

        self._bucket_buffers: list[torch.Tensor] = []
        self._param_info: list[dict[int, tuple[int, int]]] = []
        for bucket_params in self._buckets:
            total = sum(p.numel() for p in bucket_params)
            buf = torch.zeros(total, dtype=bucket_params[0].dtype, device=bucket_params[0].device)
            self._bucket_buffers.append(buf)
            offsets: dict[int, tuple[int, int]] = {}
            offset = 0
            for p in bucket_params:
                offsets[id(p)] = (offset, p.numel())
                offset += p.numel()
            self._param_info.append(offsets)

        self._param_to_bucket: dict[int, int] = {}
        for bucket_idx, bucket_params in enumerate(self._buckets):
            for p in bucket_params:
                self._param_to_bucket[id(p)] = bucket_idx

        self._bucket_pending = [len(bp) for bp in self._buckets]

        for p in grad_params:
            bucket_idx = self._param_to_bucket[id(p)]
            off, numel = self._param_info[bucket_idx][id(p)]
            p.register_post_accumulate_grad_hook(self._make_hook(bucket_idx, off, numel))

    def _make_hook(self, bucket_idx: int, offset: int, numel: int):
        def hook(p: torch.Tensor) -> None:
            self._bucket_buffers[bucket_idx][offset : offset + numel].copy_(
                p.grad.flatten()
            )
            self._bucket_pending[bucket_idx] -= 1
            if self._bucket_pending[bucket_idx] == 0:
                handle = dist.all_reduce(
                    self._bucket_buffers[bucket_idx],
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                self._handles.append((handle, bucket_idx))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, bucket_idx in self._handles:
            handle.wait()
            self._bucket_buffers[bucket_idx].div_(self._world_size)
            for p in self._buckets[bucket_idx]:
                off, n = self._param_info[bucket_idx][id(p)]
                p.grad.copy_(
                    self._bucket_buffers[bucket_idx][off : off + n].view_as(p.grad)
                )
        self._handles.clear()

    def on_train_batch_start(self):
        self._bucket_pending = [len(bp) for bp in self._buckets]
        self._handles = []


def get_ddp_individual_parameters(module: nn.Module) -> nn.Module:
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: DDPIndividualParameters, optimizer: torch.optim.Optimizer
) -> None:
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: nn.Module, bucket_size_mb: float) -> nn.Module:
    return DDPBucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(
    ddp_model: DDPBucketed, optimizer: torch.optim.Optimizer
) -> None:
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(
    ddp_model: DDPBucketed, optimizer: torch.optim.Optimizer
) -> None:
    ddp_model.on_train_batch_start()
