"""Grouped Query Attention with causal, padding and KV-cache support."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .positional_encoding import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, max_position: int,
                 theta: float = 10_000.0, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_head:
            raise ValueError("dim must be divisible by n_head")
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.rope = RotaryEmbedding(self.head_dim, max_position, theta)
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if x.ndim != 3:
            raise ValueError("attention input must have shape [batch, sequence, hidden]")
        batch, q_len, dim = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(-2)
        q = self.q_proj(x).view(batch, q_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, q_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, q_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self.rope(q, past_len)
        k = self.rope(k, past_len)
        if past_kv is not None:
            past_k, past_v = past_kv
            expected_prefix = (batch, self.n_kv_head)
            if past_k.shape[:2] != expected_prefix or past_v.shape != past_k.shape:
                raise ValueError("past_kv has incompatible batch/head dimensions")
            if past_k.size(-1) != self.head_dim:
                raise ValueError("past_kv has incompatible head dimension")
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        kv_len = k.size(-2)
        cache = (k, v) if use_cache else None

        q_positions = torch.arange(past_len, past_len + q_len, device=x.device)
        k_positions = torch.arange(kv_len, device=x.device)
        allowed = (k_positions[None, :] <= q_positions[:, None])[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, kv_len):
                raise ValueError(f"attention_mask must have shape {(batch, kv_len)}")
            allowed = allowed & attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]

        # SDPA boolean masks mark valid positions with True; explicit mask handles cached q_len != kv_len.
        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1).contiguous()
        v = v.repeat_interleave(repeat, dim=1).contiguous()
        attended = F.scaled_dot_product_attention(
            q.contiguous(), k, v, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, q_len, dim)
        return self.o_proj(attended), cache
