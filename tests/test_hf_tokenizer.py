from __future__ import annotations

import pytest

from hf_tokenizer import LyraTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> LyraTokenizer:
    return LyraTokenizer.from_pretrained()


@pytest.mark.parametrize("text", [
    "Привет, мир! Ёжик живёт в России.",
    "Hello, world! punctuation: ;!?",
    "Смешанный текст: привет / hello / print('мир')",
    "Кириллица: абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "Unicode: café — 東京 — مرحبا",
    "Эмодзи: 🧑🏽‍🚀 🌌 👍🏾",
    "Числа 0123456789, -3.14 и 1e-9.",
    "def привет(x: int) -> str:\n    return f'{x}'",
    "",
])
def test_round_trip(tokenizer: LyraTokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_and_padding(tokenizer: LyraTokenizer) -> None:
    ids = tokenizer.encode("<|user|>Привет<|turn_end|>")
    assert ids[0] == tokenizer.special_token_ids["<|user|>"]
    assert ids[-1] == tokenizer.special_token_ids["<|turn_end|>"]
    assert tokenizer.decode(ids) == "Привет"
    batch = tokenizer.encode_batch(["а", "слово длинее"], padding=True)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert 0 in batch["attention_mask"][0]
    assert batch["attention_mask"][0][0] == 0  # upstream tokenizer uses left padding


def test_vocab_contract_and_save_load(tokenizer: LyraTokenizer, tmp_path) -> None:
    assert tokenizer.vocab_size == len(tokenizer.get_vocab())
    assert tokenizer.vocab_size == 50_261
    assert tokenizer.pad_token_id == 0
    assert tokenizer.is_compatible(50_261)
    samples = ["русский текст", "function(x) { return 'мир'; }", "🙂"]
    expected = [tokenizer.encode(value) for value in samples]
    tokenizer.save(tmp_path)
    loaded = LyraTokenizer.load(tmp_path)
    assert [loaded.encode(value) for value in samples] == expected
    assert loaded.decode_batch(expected) == samples
