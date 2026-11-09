from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from config.model_config import ModelConfig
from core.generation import generate
from core.model import LyraModel


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       intermediate_size=64, block_size=16, dropout=0.0)


def test_greedy_cache_matches_full_forward() -> None:
    torch.manual_seed(3)
    model = LyraModel(tiny_config()).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    cached = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=True)
    full = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=False)
    assert cached.shape == full.shape == (1, 10)
    torch.testing.assert_close(cached, full)


def test_one_dimensional_prompt_and_eos_stop() -> None:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(block_size=16)
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, *, attention_mask=None, labels=None,
                    past_key_values=None, use_cache=False):
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), 10), -10.0,
                device=input_ids.device,
            )
            logits[:, -1, 7] = 10.0
            return {"logits": logits, "past_key_values": ("dummy",)}

    model = DummyModel().eval()
    output = generate(model, torch.tensor([1, 2]), max_new_tokens=10,
                      temperature=0.0, eos_token_id=7)
    assert output.ndim == 1
    assert output.tolist() == [1, 2, 7]


def test_multiple_eos_tokens_stop() -> None:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(block_size=16)
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, *, attention_mask=None, labels=None,
                    past_key_values=None, use_cache=False):
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), 10), -10.0,
                device=input_ids.device,
            )
            logits[:, -1, 8] = 10.0
            return {"logits": logits, "past_key_values": ("dummy",)}

    output = generate(
        DummyModel().eval(), torch.tensor([1, 2]), max_new_tokens=10,
        temperature=0.0, eos_token_id=[7, 8],
    )
    assert output.tolist() == [1, 2, 8]


def test_sampling_shape_and_repetition_penalty() -> None:
    torch.manual_seed(11)
    model = LyraModel(tiny_config()).eval()
    prompt = torch.tensor([[1, 2, 3]])
    output = generate(
        model, prompt, max_new_tokens=4, temperature=1.0, top_k=10, top_p=0.9,
        repetition_penalty=1.1,
    )
    assert output.shape == (1, 7)
    assert output.dtype == torch.long
