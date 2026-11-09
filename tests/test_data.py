from __future__ import annotations

import numpy as np
import pytest
import torch

from data.dataset import (
    CausalLMCollator,
    IGNORE_INDEX,
    PackedTokenDataset,
    SFTDataset,
    iter_conversations,
    iter_documents,
    iter_hf_documents,
    split_by_hash,
)


class StubTokenizer:
    """Small deterministic tokenizer used to test masking without HF downloads."""

    def __init__(self) -> None:
        self.special = {
            "<|user|>": 1, "<|assistant|>": 2, "<|system|>": 3, "<|turn_end|>": 4,
        }
        self.pad_token_id = 0
        self.bos_token_id = None
        self.eos_token_id = 1_000
        self._char_ids: dict[str, int] = {}
        self._next_id = 5

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for token, token_id in self.special.items():
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                character = text[index]
                if character not in self._char_ids:
                    self._char_ids[character] = self._next_id
                    self._next_id += 1
                ids.append(self._char_ids[character])
                index += 1
        return ids


def test_packed_dataset_blocks_and_labels() -> None:
    dataset = PackedTokenDataset(np.arange(20), block_size=8)
    assert len(dataset) == 2
    item = dataset[0]
    assert item["input_ids"].tolist() == list(range(8))
    assert item["labels"].tolist() == item["input_ids"].tolist()
    assert item["attention_mask"].tolist() == [1] * 8


def test_sft_dataset_masks_prompt_and_trains_assistant() -> None:
    tokenizer = StubTokenizer()
    conversation = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    dataset = SFTDataset([conversation], tokenizer, block_size=128)
    item = dataset[0]
    labels = item["labels"].tolist()
    ids = item["input_ids"].tolist()
    assert len(ids) == len(labels)
    assert labels[0] == IGNORE_INDEX
    assert labels[-1] == 4  # assistant turn_end is trained
    assistant_content_id = tokenizer.encode("A")[0]
    assert assistant_content_id in ids
    trained = [token for token in labels if token != IGNORE_INDEX]
    assert trained[-2:] == [assistant_content_id, 4]
    assert labels.count(IGNORE_INDEX) > len(trained)


def test_collator_pads_right_and_masks_labels() -> None:
    tokenizer = StubTokenizer()
    first = {
        "input_ids": torch.tensor([5, 6, 7]),
        "labels": torch.tensor([5, 6, 7]),
    }
    second = {
        "input_ids": torch.tensor([8, 9, 10, 11, 12]),
        "labels": torch.tensor([8, 9, 10, 11, 12]),
    }
    batch = CausalLMCollator(tokenizer.pad_token_id)([first, second])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [5, 6, 7, IGNORE_INDEX, IGNORE_INDEX]


def test_hash_split_is_deterministic_and_disjoint() -> None:
    documents = [f"document {index}" for index in range(200)]
    train_a, val_a = split_by_hash(documents, 0.2, seed=7)
    train_b, val_b = split_by_hash(documents, 0.2, seed=7)
    assert train_a == train_b and val_a == val_b
    assert val_a and not set(train_a) & set(val_a)
    assert set(train_a) | set(val_a) == set(documents)


def test_local_hf_dataset_loader(tmp_path) -> None:
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict({"text": ["a", "b", "c"]})
    dataset.save_to_disk(str(tmp_path / "hf"))
    loaded = list(iter_hf_documents(
        str(tmp_path / "hf"), split="train", streaming=False, text_field="text",
    ))
    assert loaded == ["a", "b", "c"]


def test_file_loaders(tmp_path) -> None:
    text_path = tmp_path / "docs.txt"
    text_path.write_text("первый\n\nвторой\n", encoding="utf-8")
    assert list(iter_documents(text_path)) == ["первый", "второй"]

    jsonl_path = tmp_path / "chat.jsonl"
    jsonl_path.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]}\n'
        '{"prompt":"q","response":"a"}\n',
        encoding="utf-8",
    )
    conversations = list(iter_conversations(jsonl_path))
    assert len(conversations) == 2
    assert conversations[0][1] == {"role": "assistant", "content": "ok"}
    assert list(iter_documents(jsonl_path)) == ["hi\nok", "q\na"]
