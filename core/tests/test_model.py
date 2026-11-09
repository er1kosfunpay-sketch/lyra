from __future__ import annotations

import torch

from config.model_config import ModelConfig
from core.model import LyraModel


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       intermediate_size=64, block_size=16, dropout=0.0)


def test_forward_backward_tying_and_gqa_components() -> None:
    model = LyraModel(tiny_config())
    ids = torch.randint(0, 300, (2, 8))
    result = model(ids, labels=ids)
    assert result["logits"].shape == (2, 8, 300)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()
    attention = model.layers[0].attn
    assert attention.n_head == 4 and attention.n_kv_head == 2
    assert attention.k_proj.out_features == 16
    assert hasattr(model.layers[0].ffn, "gate_proj")


def test_causal_attention() -> None:
    torch.manual_seed(12)
    model = LyraModel(tiny_config()).eval()
    first = torch.tensor([[5, 6, 7, 8]])
    second = torch.tensor([[5, 6, 99, 101]])
    with torch.no_grad():
        left, right = model(first)["logits"], model(second)["logits"]
    torch.testing.assert_close(left[:, :2], right[:, :2], rtol=0, atol=1e-6)


def test_cache_matches_full_sequence() -> None:
    torch.manual_seed(27)
    model = LyraModel(tiny_config()).eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        full = model(ids)["logits"]
        prefill = model(ids[:, :3], use_cache=True)
        last = model(ids[:, 3:], past_key_values=prefill["past_key_values"], use_cache=True)
    torch.testing.assert_close(full[:, 3:], last["logits"], rtol=1e-5, atol=1e-6)


def test_invalid_gqa_dimensions_rejected() -> None:
    try:
        ModelConfig(vocab_size=300, n_head=6, n_kv_head=4, n_embd=32)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("invalid GQA dimensions were accepted")


def test_exact_parameter_count() -> None:
    model = LyraModel(tiny_config())
    # 9,600 embedding + 2*(3,072 GQA + 6,144 SwiGLU + 64 RMSNorm) + 32 final norm.
    assert model.num_parameters() == 28_192


def test_input_and_cache_validation() -> None:
    model = LyraModel(tiny_config())
    with torch.no_grad():
        try:
            model(torch.tensor([[300, 1]]))
        except ValueError as exc:
            assert "vocabulary" in str(exc)
        else:
            raise AssertionError("out of range token ID accepted")
        try:
            model(torch.ones((1, 2), dtype=torch.long), attention_mask=torch.ones((1, 1)))
        except ValueError as exc:
            assert "attention_mask" in str(exc)
        else:
            raise AssertionError("invalid attention mask accepted")


def test_context_limit_includes_cached_tokens() -> None:
    model = LyraModel(tiny_config())
    ids = torch.ones((1, 16), dtype=torch.long)
    cache = model(ids, use_cache=True)["past_key_values"]
    try:
        model(torch.ones((1, 1), dtype=torch.long), past_key_values=cache)
    except ValueError as exc:
        assert "context" in str(exc)
    else:
        raise AssertionError("cached context overflow accepted")
