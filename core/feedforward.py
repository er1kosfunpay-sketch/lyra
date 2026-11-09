"""SwiGLU feed-forward network."""

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))
