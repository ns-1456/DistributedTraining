from __future__ import annotations

import torch

try:
    import flash_attention_cuda
except ImportError:
    flash_attention_cuda = None


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        assert flash_attention_cuda is not None, (
            "flash_attention_cuda not built. Run: cd cuda && pip install -e ."
        )
        O, L = flash_attention_cuda.forward(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dO: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_cuda.backward(Q, K, V, O, dO.contiguous(), L, ctx.is_causal)
        return dQ, dK, dV, None


def flash_attention_2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    return FlashAttention2.apply(Q, K, V, is_causal)
