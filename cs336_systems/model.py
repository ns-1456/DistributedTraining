from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    max_seq_len: int = 1024


CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def _trunc_normal_(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)
    return tensor


def _init_linear(weight: torch.Tensor) -> None:
    d_in, d_out = weight.shape
    _trunc_normal_(weight, 0.0, math.sqrt(2.0 / (d_in + d_out)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached: int = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        self._build_cache(seq_len)
        assert self._cos_cached is not None and self._sin_cached is not None

        cos = self._cos_cached[:seq_len].to(q.dtype)
        sin = self._sin_cached[:seq_len].to(q.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)
    return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_flash: bool = False) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_flash = use_flash

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        _init_linear(self.W_q)
        _init_linear(self.W_k)
        _init_linear(self.W_v)
        _init_linear(self.W_o)

        self.rope = RotaryPositionEmbedding(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        Q, K = self.rope(Q, K)

        if self.use_flash:
            try:
                from cuda.flash_attn_func import flash_attention_2
                attn_out = flash_attention_2(Q, K, V, is_causal=True)
            except ImportError:
                from cs336_systems.flash_attention import flash_attention_pytorch
                BH = B * self.num_heads
                q_flat = Q.reshape(BH, T, self.d_head)
                k_flat = K.reshape(BH, T, self.d_head)
                v_flat = V.reshape(BH, T, self.d_head)
                attn_out = flash_attention_pytorch(q_flat, k_flat, v_flat, is_causal=True)
                attn_out = attn_out.reshape(B, self.num_heads, T, self.d_head)
        else:
            attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return torch.matmul(attn_out, self.W_o)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W3 = nn.Parameter(torch.empty(d_model, d_ff))

        _init_linear(self.W1)
        _init_linear(self.W2)
        _init_linear(self.W3)

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._silu(torch.matmul(x, self.W1))
        up = torch.matmul(x, self.W3)
        return torch.matmul(gate * up, self.W2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_flash: bool = False) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, use_flash=use_flash)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Parameter(torch.empty(vocab_size, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, use_flash=use_flash) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Parameter(torch.empty(d_model, vocab_size))

        _trunc_normal_(self.embedding, 0.0, 1.0)
        _init_linear(self.head)

        causal = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embedding[input_ids]

        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        return torch.matmul(x, self.head)

    @staticmethod
    def from_config(config_name: str, use_flash: bool = False) -> "TransformerLM":
        cfg = CONFIGS[config_name]
        return TransformerLM(
            vocab_size=10000,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            max_seq_len=1024,
            use_flash=use_flash,
        )
