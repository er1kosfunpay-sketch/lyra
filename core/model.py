"""Lyra decoder-only Transformer with randomly initialized, project-owned weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from config.model_config import ModelConfig
from .normalization import RMSNorm
from .transformer import TransformerBlock


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, *,
                    ignore_index: int = -100, chunk_size: int = 128) -> torch.Tensor:
    """Shifted next-token loss with FP32 chunked cross-entropy."""
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    total = logits.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, shift_labels.size(1), chunk_size):
        end = min(start + chunk_size, shift_labels.size(1))
        chunk_logits = shift_logits[:, start:end].float()
        chunk_labels = shift_labels[:, start:end].long()
        valid = chunk_labels.ne(ignore_index)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        total = total + F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        count += n_valid
    if count == 0:
        # Keep the loss connected to the graph so backward remains well-defined.
        return logits.flatten()[0].float() * 0.0
    return total / count


class LyraModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.layers = nn.ModuleList(TransformerBlock(self.config) for _ in range(self.config.n_layer))
        self.final_norm = RMSNorm(self.config.n_embd, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # Residual projections use depth-scaled initialization from the GPT-2 recipe.
        residual_std = self.config.init_std / (2 * self.config.n_layer) ** 0.5
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=residual_std)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        # The tied output projection shares storage with token_embeddings; do not
        # initialize the same tensor twice through two module names.
        if self.config.tie_embeddings and module is self.lm_head:
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade compute for activation memory during training."""
        self.gradient_checkpointing = enabled

    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
                use_cache: bool = False) -> dict[str, torch.Tensor | tuple]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must have an integer dtype")
        batch, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids cannot be empty")
        # Avoid GPU synchronization on every forward; embedding performs the same bounds check on CUDA.
        if input_ids.device.type == "cpu" and (input_ids.min().item() < 0 or
                                                 input_ids.max().item() >= self.config.vocab_size):
            raise ValueError("input_ids contains an ID outside the model vocabulary")
        if past_key_values is not None and len(past_key_values) != self.config.n_layer:
            raise ValueError("past_key_values must contain one key/value pair per layer")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if past_key_values is not None and any(k.size(-2) != past_len or v.size(-2) != past_len
                                               for k, v in past_key_values):
            raise ValueError("All cached layers must have the same sequence length")
        if past_len + seq_len > self.config.block_size:
            raise ValueError("Input plus cached sequence exceeds configured context length")
        total_len = past_len + seq_len
        if attention_mask is not None and tuple(attention_mask.shape) != (batch, total_len):
            raise ValueError(f"attention_mask must have shape {(batch, total_len)}")
        if labels is not None:
            if tuple(labels.shape) != (batch, seq_len):
                raise ValueError("labels must have the same [batch, sequence] shape as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must have an integer dtype")

        use_checkpoint = (self.gradient_checkpointing and self.training
                          and past_key_values is None and not use_cache)
        x = self.token_embeddings(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            if use_checkpoint:
                x, present = activation_checkpoint(
                    layer, x, attention_mask, None, False, use_reentrant=False,
                )
            else:
                x, present = layer(x, attention_mask, past, use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.final_norm(x))
        result: dict[str, torch.Tensor | tuple] = {"logits": logits}
        if labels is not None:
            if seq_len < 2:
                raise ValueError("At least two tokens are required to compute next-token loss")
            result["loss"] = _causal_lm_loss(
                logits, labels, chunk_size=self.config.loss_chunk_size,
            )
        if use_cache:
            result["past_key_values"] = tuple(presents)
        return result

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if not trainable_only or parameter.requires_grad)

    def parameter_summary(self) -> dict[str, int]:
        summary = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        summary["total"] = self.num_parameters()
        summary["trainable"] = self.num_parameters(trainable_only=True)
        return summary

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Autoregressive generation; see :func:`core.generation.generate`."""
        from .generation import generate

        return generate(self, input_ids, **kwargs)
