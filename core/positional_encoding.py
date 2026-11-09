"""Rotary positional embeddings (RoPE)."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim]
        length = x.size(-2)
        end = position_offset + length
        if position_offset < 0 or end > self.cos.size(0):
            raise ValueError(f"RoPE position range [{position_offset}, {end}) exceeds context {self.cos.size(0)}")
        cos = self.cos[position_offset:end].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        sin = self.sin[position_offset:end].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
