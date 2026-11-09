"""Dataset, tokenization and batching utilities for Lyra training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".json"}


def _iter_files(path: str | Path) -> Iterator[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported data file extension: {root.suffix}")
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Data path does not exist: {root}")
    found = False
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            found = True
            yield candidate
    if not found:
        raise FileNotFoundError(f"No .txt/.jsonl/.json files found under {root}")


def _json_rows(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict) and isinstance(value.get("data"), list):
            yield from value["data"]
        else:
            yield value


def _document_from_row(row: Any) -> str | None:
    if isinstance(row, str):
        return row
    if isinstance(row, Mapping):
        for key in ("text", "content", "document", "code", "raw"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        if isinstance(row.get("messages"), list):
            return "\n".join(
                str(message.get("content", "")) for message in row["messages"]
                if isinstance(message, Mapping)
            )
        if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
            return f"{row['prompt']}\n{row['response']}"
    return None


def iter_documents(path: str | Path, *, max_docs: int | None = None) -> Iterator[str]:
    """Yield plain-text documents from .txt, .jsonl or .json sources."""
    count = 0
    for file in _iter_files(path):
        if file.suffix.lower() == ".txt":
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    yield text
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        return
            continue
        for row in _json_rows(file):
            text = _document_from_row(row)
            if not text or not text.strip():
                continue
            yield text
            count += 1
            if max_docs is not None and count >= max_docs:
                return


ROLE_ALIASES = {
    "user": "user", "human": "user", "prompt": "user", "question": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "system": "system", "instruction": "system",
}


def _normalise_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for message in value:
        if isinstance(message, str):
            role, content = ("user", message) if len(messages) % 2 == 0 else ("assistant", message)
        elif isinstance(message, Mapping):
            role = ROLE_ALIASES.get(str(message.get("role", message.get("from", ""))).lower())
            content = message.get("content", message.get("value", message.get("text")))
            if role is None or not isinstance(content, str):
                continue
        else:
            continue
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages or None


def _conversation_from_row(row: Any) -> list[dict[str, str]] | None:
    if isinstance(row, list):
        return _normalise_messages(row)
    if not isinstance(row, Mapping):
        return None
    for key in ("messages", "conversation", "conversations"):
        if key in row:
            messages = _normalise_messages(row[key])
            if messages:
                return messages
    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        return _normalise_messages([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ])
    if isinstance(row.get("question"), str) and isinstance(row.get("answer"), str):
        return _normalise_messages([
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ])
    if isinstance(row.get("instruction"), str):
        user = row["instruction"]
        if isinstance(row.get("input"), str) and row["input"].strip():
            user = f"{user}\n\n{row['input']}"
        output = row.get("output", row.get("response"))
        if isinstance(output, str):
            return _normalise_messages([
                {"role": "user", "content": user},
                {"role": "assistant", "content": output},
            ])
    return None


def iter_conversations(path: str | Path, *,
                       max_docs: int | None = None) -> Iterator[list[dict[str, str]]]:
    """Yield conversations from JSON/JSONL sources.

    Supported shapes: ``{"messages": [...]}``, ``{"conversation": [...]}``,
    ``{"prompt": ..., "response": ...}`` and Alpaca-style instruction rows.
    """
    count = 0
    for file in _iter_files(path):
        if file.suffix.lower() == ".txt":
            raise ValueError("SFT mode requires .jsonl or .json; plain text has no roles")
        for row in _json_rows(file):
            conversation = _conversation_from_row(row)
            if not conversation:
                continue
            yield conversation
            count += 1
            if max_docs is not None and count >= max_docs:
                return


def _load_hf_dataset(dataset_name: str, config_name: str | None, split: str,
                     streaming: bool):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install the optional 'datasets' package to use --hf-dataset"
        ) from exc
    local = Path(dataset_name)
    if local.exists():
        dataset = load_from_disk(str(local))
        if hasattr(dataset, "keys"):
            if split not in dataset:
                raise ValueError(f"Split {split!r} is absent from {local}")
            return dataset[split]
        return dataset
    return load_dataset(dataset_name, config_name, split=split, streaming=streaming)


def iter_hf_documents(dataset_name: str, *, config_name: str | None = None,
                      split: str = "train", streaming: bool = True,
                      text_field: str | None = None,
                      max_docs: int | None = None) -> Iterator[str]:
    """Stream plain-text documents from a Hugging Face dataset."""
    dataset = _load_hf_dataset(dataset_name, config_name, split, streaming)
    count = 0
    for row in dataset:
        if text_field:
            text = row.get(text_field) if hasattr(row, "get") else None
        else:
            text = _document_from_row(row)
        if not isinstance(text, str) or not text.strip():
            continue
        yield text
        count += 1
        if max_docs is not None and count >= max_docs:
            return


def iter_hf_conversations(dataset_name: str, *, config_name: str | None = None,
                          split: str = "train", streaming: bool = True,
                          max_docs: int | None = None) -> Iterator[list[dict[str, str]]]:
    """Stream role-tagged conversations from a Hugging Face dataset."""
    dataset = _load_hf_dataset(dataset_name, config_name, split, streaming)
    count = 0
    for row in dataset:
        conversation = _conversation_from_row(row)
        if not conversation:
            continue
        yield conversation
        count += 1
        if max_docs is not None and count >= max_docs:
            return


def _hash_bucket(item: Any, seed: int, key: Callable[[Any], str] | None) -> float:
    if key is not None:
        text = key(item)
    elif isinstance(item, str):
        text = item
    else:
        text = json.dumps(item, ensure_ascii=False, sort_keys=True)
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


def split_by_hash(items: Iterable[Any], val_ratio: float, *, seed: int = 42,
                  key: Callable[[Any], str] | None = None) -> tuple[list[Any], list[Any]]:
    """Deterministically split documents before windowing to avoid leakage."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    materialised = list(items)
    train: list[Any] = []
    validation: list[Any] = []
    for item in materialised:
        (validation if _hash_bucket(item, seed, key) < val_ratio else train).append(item)
    if val_ratio > 0 and not validation and len(materialised) > 1:
        validation.append(train.pop())
    return train, validation


class PackedTokenDataset(Dataset):
    """Fixed-length next-token blocks over a flat token stream."""

    def __init__(self, tokens: Sequence[int] | np.ndarray | torch.Tensor,
                 block_size: int, *, drop_last: bool = True) -> None:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        self.tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        self.block_size = int(block_size)
        self.drop_last = drop_last
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.tokens.size == 0:
            raise ValueError("token stream is empty")
        full_blocks, remainder = divmod(self.tokens.size, self.block_size)
        self.n_blocks = full_blocks + (0 if drop_last or remainder == 0 else 1)
        if self.n_blocks == 0:
            raise ValueError("token stream is shorter than one block")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        block = self.tokens[start:start + self.block_size]
        input_ids = torch.from_numpy(block.copy())
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


class SFTDataset(Dataset):
    """Conversation dataset with loss masked to assistant responses."""

    def __init__(self, conversations: Iterable[Sequence[dict[str, str]]], tokenizer: Any,
                 block_size: int, *, add_bos: bool = False, add_eos: bool = False) -> None:
        self.block_size = int(block_size)
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.examples: list[tuple[list[int], list[int]]] = []
        for conversation in conversations:
            encoded = self._encode(conversation, tokenizer, add_bos=add_bos, add_eos=add_eos)
            if encoded is None:
                continue
            ids, labels = encoded
            ids, labels = ids[:self.block_size], labels[:self.block_size]
            if len(ids) < 2 or all(label == IGNORE_INDEX for label in labels):
                continue
            self.examples.append((ids, labels))
        if not self.examples:
            raise ValueError("No usable SFT examples were found")

    @staticmethod
    def _encode(conversation: Sequence[dict[str, str]], tokenizer: Any, *,
                add_bos: bool, add_eos: bool) -> tuple[list[int], list[int]] | None:
        ids: list[int] = []
        labels: list[int] = []

        def add_text(text: str, train: bool) -> None:
            try:
                piece = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
            except TypeError:
                piece = [int(token) for token in tokenizer.encode(text)]
            ids.extend(piece)
            labels.extend(piece if train else [IGNORE_INDEX] * len(piece))

        if add_bos and getattr(tokenizer, "bos_token_id", None) is not None:
            ids.append(int(tokenizer.bos_token_id))
            labels.append(IGNORE_INDEX)
        for message in conversation:
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                continue
            add_text(f"<|{role}|>", train=False)
            is_assistant = role == "assistant"
            add_text(content, train=is_assistant)
            add_text("<|turn_end|>", train=is_assistant)
        if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
            ids.append(int(tokenizer.eos_token_id))
            labels.append(int(tokenizer.eos_token_id))
        return (ids, labels) if ids else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids, labels = self.examples[index]
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones_like(input_ids),
        }


class CausalLMCollator:
    """Right-pad batches and mask padding from the loss."""

    def __init__(self, pad_token_id: int, *, pad_to_multiple_of: int | None = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(int(example["input_ids"].numel()) for example in examples)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        input_ids = torch.full((len(examples), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(examples), max_length), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
        for row, example in enumerate(examples):
            length = int(example["input_ids"].numel())
            input_ids[row, :length] = example["input_ids"]
            labels[row, :length] = example["labels"]
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def tokenize_documents(documents: Iterable[str], tokenizer: Any, *, add_eos: bool = True,
                       max_tokens: int | None = None, batch_size: int = 256) -> np.ndarray:
    """Tokenize documents and concatenate them into one flat token stream."""
    flat: list[int] = []
    pending: list[str] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def flush() -> None:
        if not pending:
            return
        if hasattr(tokenizer, "encode_batch"):
            rows = tokenizer.encode_batch(pending, padding=False, truncation=False)["input_ids"]
        else:
            rows = [tokenizer.encode(text) for text in pending]
        for row in rows:
            flat.extend(int(token) for token in row)
            if add_eos and eos_token_id is not None:
                flat.append(int(eos_token_id))
        pending.clear()

    for document in documents:
        if not document or not document.strip():
            continue
        pending.append(document)
        if len(pending) >= batch_size:
            flush()
        if max_tokens is not None and len(flat) >= max_tokens:
            break
    flush()
    array = np.asarray(flat, dtype=np.int64)
    if max_tokens is not None:
        array = array[:max_tokens]
    return array
