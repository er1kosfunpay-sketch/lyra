"""Pre-norm decoder Transformer block."""

import torch
from torch import nn

from config.model_config import ModelConfig
from .attention import GroupedQueryAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config.n_embd, config.n_head, config.n_kv_head,
            config.block_size, config.rope_theta, config.bias, config.dropout)
        self.ffn_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.ffn = SwiGLU(config.n_embd, config.intermediate_size, config.bias, config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention, cache = self.attn(self.attn_norm(x), attention_mask, past_kv, use_cache)
        x = x + self.resid_dropout(attention)
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x, cache
