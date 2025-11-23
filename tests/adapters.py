from __future__ import annotations

from typing import Type

import torch

from cs336_systems.flash_attention import FlashAttentionPyTorch
from cs336_systems.ddp_custom import (
    DDPIndividualParameters,
    DDPBucketed,
    get_ddp_individual_parameters as _get_ddp_individual,
    ddp_individual_parameters_on_after_backward as _ddp_individual_after_bwd,
    get_ddp_bucketed as _get_ddp_bucketed,
    ddp_bucketed_on_after_backward as _ddp_bucketed_after_bwd,
    ddp_bucketed_on_train_batch_start as _ddp_bucketed_train_start,
)
from cs336_systems.optimizer_sharding import (
    get_sharded_optimizer as _get_sharded_optimizer,
)


def get_flashattention_autograd_function_pytorch() -> Type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> Type:
    try:
        from cuda.flash_attn_func import FlashAttention2
        return FlashAttention2
    except ImportError:
        raise NotImplementedError(
            "Triton/CUDA FlashAttention not available. "
            "Build the CUDA extension: cd cuda && pip install -e ."
        )


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return _get_ddp_individual(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    _ddp_individual_after_bwd(ddp_model, optimizer)


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    return _get_ddp_bucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    _ddp_bucketed_after_bwd(ddp_model, optimizer)


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    _ddp_bucketed_train_start(ddp_model, optimizer)


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    return _get_sharded_optimizer(params, optimizer_cls, **kwargs)
