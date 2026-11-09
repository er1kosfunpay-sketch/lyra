"""Autoregressive sampling with KV-cache support."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor,
                              penalty: float) -> None:
    if penalty == 1.0:
        return
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    for row in range(input_ids.size(0)):
        tokens = torch.unique(input_ids[row])
        values = logits[row, tokens]
        logits[row, tokens] = torch.where(values > 0, values / penalty, values * penalty)


def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_sorted = cumulative > top_p
    # Always keep the most likely token.
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted, dtype=torch.bool).scatter(
        1, sorted_indices, remove_sorted,
    )
    return logits.masked_fill(remove, float("-inf"))


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, *, max_new_tokens: int = 64,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, repetition_penalty: float = 1.0,
             eos_token_id: int | Sequence[int] | None = None, pad_token_id: int | None = None,
             attention_mask: torch.Tensor | None = None,
             use_cache: bool = True) -> torch.Tensor:
    """Generate tokens for a batch of prompts.

    ``input_ids`` may be one- or two-dimensional. The returned tensor keeps the
    same rank and includes the prompt tokens. Sampling is greedy when
    ``temperature <= 0``.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    was_1d = input_ids.ndim == 1
    if was_1d:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.size(1) == 0:
        raise ValueError("input_ids cannot be empty")
    if max_new_tokens == 0:
        return input_ids.squeeze(0) if was_1d else input_ids
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("attention_mask must match input_ids shape")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    block_size = int(model.config.block_size)
    if input_ids.size(1) > block_size:
        raise ValueError("prompt is longer than the model context length")

    eos_ids: torch.Tensor | None = None
    if eos_token_id is not None:
        values = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        if values:
            eos_ids = torch.tensor(values, dtype=torch.long, device=input_ids.device)

    previous_training = model.training
    model.eval()
    batch = input_ids.size(0)
    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    past_key_values = None
    try:
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= block_size:
                break
            if use_cache and past_key_values is not None:
                step_ids = input_ids[:, -1:]
                outputs = model(
                    step_ids, attention_mask=attention_mask,
                    past_key_values=past_key_values, use_cache=True,
                )
            else:
                outputs = model(
                    input_ids, attention_mask=attention_mask,
                    past_key_values=past_key_values if use_cache else None,
                    use_cache=use_cache,
                )
            logits = outputs["logits"][:, -1, :].float()
            if use_cache:
                past_key_values = outputs["past_key_values"]

            _apply_repetition_penalty(logits, input_ids, repetition_penalty)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    logits = _filter_top_k(logits, top_k)
                if top_p is not None:
                    logits = _filter_top_p(logits, top_p)
                probabilities = F.softmax(logits, dim=-1)
                if not torch.isfinite(probabilities).all() or probabilities.sum(dim=-1).min() <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probabilities, num_samples=1)

            if eos_ids is not None:
                already_finished = finished.clone()
                finished |= torch.isin(next_token.squeeze(1), eos_ids)
                # Keep rows that finished on earlier steps on the first EOS while the
                # rest of the batch continues. Newly finished rows keep their own token.
                if already_finished.any():
                    next_token = next_token.masked_fill(already_finished.unsqueeze(1), int(eos_ids[0]))
                if finished.all():
                    input_ids = torch.cat((input_ids, next_token), dim=1)
                    attention_mask = torch.cat(
                        (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1,
                    )
                    break

            input_ids = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1,
            )
            if not use_cache:
                past_key_values = None
    finally:
        model.train(previous_training)

    return input_ids.squeeze(0) if was_1d else input_ids
