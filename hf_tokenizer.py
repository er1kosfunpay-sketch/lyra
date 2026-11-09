"""Adapter for the Russian GPT-3 byte-level BPE tokenizer hosted on Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_ID = "ai-forever/rugpt3small_based_on_gpt2"
DEFAULT_TOKENIZER_REVISION = "a9307e696cd3c5b7f953ff4cb19d76a4d81821d5"
ROLE_TOKENS = ("<|user|>", "<|assistant|>", "<|system|>", "<|turn_end|>")


class LyraTokenizer:
    """HF tokenizer adapter. Only tokenizer assets are downloaded, never model weights."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, source: str, revision: str | None) -> None:
        self.tokenizer = tokenizer
        self.source = source
        self.revision = revision or tokenizer.init_kwargs.get("_commit_hash")

    @classmethod
    def from_pretrained(cls, tokenizer_id: str = DEFAULT_TOKENIZER_ID,
                        *, revision: str | None = DEFAULT_TOKENIZER_REVISION,
                        cache_dir: str | Path | None = None, local_files_only: bool = False) -> "LyraTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, revision=revision, cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only, trust_remote_code=False, use_fast=True,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": list(ROLE_TOKENS)})
        return cls(tokenizer, tokenizer_id, revision)

    @classmethod
    def load(cls, directory: str | Path) -> "LyraTokenizer":
        path = Path(directory)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, use_fast=True)
        meta_path = path / "lyra_tokenizer.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if not set(ROLE_TOKENS).issubset(tokenizer.all_special_tokens):
            raise ValueError("Saved tokenizer is missing Lyra role tokens")
        return cls(tokenizer, meta.get("source", str(path)), meta.get("revision"))

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        metadata = {
            "source": self.source, "revision": self.revision, "vocab_size": self.vocab_size,
            "role_tokens": self.special_token_ids, "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id, "eos_token_id": self.eos_token_id,
        }
        (path / "lyra_tokenizer.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is None:
            raise ValueError("HF tokenizer has no padding token")
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {token: int(self.tokenizer.convert_tokens_to_ids(token)) for token in ROLE_TOKENS}

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos and self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        if add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=False)

    def encode_batch(self, texts: list[str], *, padding: bool = False, truncation: bool = False,
                     max_length: int | None = None) -> dict[str, Any]:
        return self.tokenizer(texts, add_special_tokens=False, padding=padding, truncation=truncation,
                              max_length=max_length, return_attention_mask=True)

    def decode_batch(self, rows: list[list[int]], *, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(rows, skip_special_tokens=skip_special_tokens,
                                           clean_up_tokenization_spaces=False)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_special_tokens(self) -> dict[str, int]:
        ids = dict(self.special_token_ids)
        for name, value in (("pad", self.pad_token_id), ("bos", self.bos_token_id), ("eos", self.eos_token_id)):
            if value is not None:
                ids[name] = int(value)
        return ids

    def is_compatible(self, model_vocab_size: int) -> bool:
        return model_vocab_size == self.vocab_size
