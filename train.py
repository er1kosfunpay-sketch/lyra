"""Command-line training entry point for Lyra."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from core.model import LyraModel
from data.dataset import (
    CausalLMCollator,
    PackedTokenDataset,
    SFTDataset,
    iter_conversations,
    iter_documents,
    iter_hf_conversations,
    iter_hf_documents,
    split_by_hash,
    tokenize_documents,
)
from hf_tokenizer import DEFAULT_TOKENIZER_ID, DEFAULT_TOKENIZER_REVISION, LyraTokenizer
from training.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lyra from scratch.")
    parser.add_argument("--data", type=Path, help="Training corpus (.txt/.jsonl/.json)")
    parser.add_argument("--val-data", type=Path, help="Optional held-out corpus")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset name for streaming")
    parser.add_argument("--hf-config", help="Hugging Face dataset configuration/subset")
    parser.add_argument("--hf-split", default=None, help="Hugging Face split (default: train)")
    parser.add_argument("--hf-text-field", help="Field containing text (e.g. 'text')")
    parser.add_argument("--no-hf-streaming", dest="hf_streaming", action="store_false",
                        default=None, help="Download the HF dataset instead of streaming")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--preset", choices=["base", "tiny"], default="base")
    parser.add_argument("--resume", type=Path, help="Checkpoint file or run directory")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--train-tokens", type=Path, help="Pre-tokenized .npy token stream")
    parser.add_argument("--val-tokens", type=Path, help="Pre-tokenized .npy validation stream")
    parser.add_argument("--save-tokens", action="store_true",
                        help="Cache tokenized pretraining streams as .npy")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny offline end-to-end training check")

    for name in (
        "max_steps", "batch_size", "grad_accum_steps", "lr", "min_lr_ratio",
        "warmup_steps", "weight_decay", "grad_clip", "log_interval",
        "eval_interval", "eval_batches", "save_interval", "checkpoint_history",
        "num_workers",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=_arg_type(name))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--compile", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", default=None)
    return parser.parse_args(argv)


def _arg_type(name: str):
    if name in {"lr", "min_lr_ratio", "weight_decay", "grad_clip"}:
        return float
    return int


def _resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    return path / "latest.pt" if path.is_dir() else path


def _path_str(value: str | Path | None) -> str | None:
    return str(Path(value).resolve()) if value is not None else None


def _peek_checkpoint(path: str | Path) -> tuple[ModelConfig, dict[str, Any]]:
    state = torch.load(_resolve_checkpoint(path), map_location="cpu", weights_only=False, mmap=True)
    model_config = ModelConfig(**state["model_config"])
    train_config = dict(state.get("train_config", {}))
    return model_config, train_config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(args: argparse.Namespace) -> LyraTokenizer:
    if args.tokenizer_dir:
        return LyraTokenizer.load(args.tokenizer_dir)
    return LyraTokenizer.from_pretrained(
        args.tokenizer_id, revision=args.tokenizer_revision,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )


def _build_model_config(args: argparse.Namespace, tokenizer: LyraTokenizer | None,
                        resumed: ModelConfig | None) -> ModelConfig:
    if resumed is not None:
        config = resumed
    elif args.model_config:
        config = ModelConfig.load(args.model_config)
    elif args.preset == "tiny":
        config = ModelConfig(
            vocab_size=512, n_layer=2, n_head=4, n_kv_head=2, n_embd=64,
            intermediate_size=128, block_size=64, dropout=0.0,
        )
    else:
        config = ModelConfig()
    if tokenizer is not None:
        if resumed is not None and tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab ({tokenizer.vocab_size}) does not match checkpoint "
                f"vocab ({config.vocab_size})"
            )
        config.vocab_size = tokenizer.vocab_size
    return config


def _smoke_datasets(config: ModelConfig, seed: int) -> tuple[PackedTokenDataset, PackedTokenDataset, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_tokens = rng.integers(0, config.vocab_size, size=config.block_size * 64, dtype=np.int64)
    val_tokens = rng.integers(0, config.vocab_size, size=config.block_size * 16, dtype=np.int64)
    train_dataset = PackedTokenDataset(train_tokens, config.block_size)
    val_dataset = PackedTokenDataset(val_tokens, config.block_size)
    manifest = {
        "mode": "smoke", "train_tokens": int(train_tokens.size),
        "val_tokens": int(val_tokens.size), "block_size": config.block_size,
        "vocab_size": config.vocab_size, "seed": seed, "val_ratio": None,
        "max_docs": None, "max_tokens": None, "hf_streaming": None,
    }
    return train_dataset, val_dataset, manifest


def _build_datasets(args: argparse.Namespace, tokenizer: LyraTokenizer,
                    config: ModelConfig) -> tuple[Any, Any, dict[str, Any]]:
    if args.smoke_test:
        return _smoke_datasets(config, args.seed)
    if args.data is not None and args.hf_dataset:
        raise SystemExit("Use either --data or --hf-dataset, not both")
    if args.train_tokens and (args.data is not None or args.hf_dataset):
        raise SystemExit("Use either --train-tokens or a text source, not both")
    if args.data is None and not args.hf_dataset and not args.train_tokens:
        raise SystemExit(
            "--data, --hf-dataset or --train-tokens is required unless --smoke-test is used"
        )
    if args.hf_dataset and args.hf_streaming and args.max_docs is None:
        raise SystemExit("--max-docs is required for streaming --hf-dataset")
    if args.train_tokens and args.mode != "pretrain":
        raise SystemExit("--train-tokens is only supported in pretrain mode")

    if args.mode == "pretrain" and args.train_tokens:
        train_tokens = np.load(args.train_tokens, mmap_mode="r")
        val_tokens = (
            np.load(args.val_tokens, mmap_mode="r")
            if args.val_tokens else np.empty(0, dtype=np.int64)
        )
        train_dataset = PackedTokenDataset(train_tokens, config.block_size)
        val_dataset = (
            PackedTokenDataset(val_tokens, config.block_size)
            if val_tokens.size >= config.block_size else None
        )
        manifest = {
            "mode": "pretrain", "data": None, "val_data": None,
            "train_tokens_file": _path_str(args.train_tokens),
            "val_tokens_file": _path_str(args.val_tokens),
            "train_tokens": int(train_tokens.size), "val_tokens": int(val_tokens.size),
            "block_size": config.block_size, "vocab_size": config.vocab_size,
            "seed": args.seed, "val_ratio": args.val_ratio,
            "max_docs": args.max_docs, "max_tokens": args.max_tokens,
            "hf_dataset": None, "hf_config": None, "hf_split": None,
            "hf_text_field": None, "hf_streaming": None,
        }
        return train_dataset, val_dataset, manifest

    if args.mode == "pretrain":
        if args.hf_dataset:
            documents = list(iter_hf_documents(
                args.hf_dataset, config_name=args.hf_config, split=args.hf_split or "train",
                streaming=args.hf_streaming, text_field=args.hf_text_field,
                max_docs=args.max_docs,
            ))
        else:
            documents = list(iter_documents(args.data, max_docs=args.max_docs))
        if not documents:
            raise SystemExit("No documents found in the training source")
        if args.val_data:
            train_documents = documents
            val_documents = list(iter_documents(args.val_data, max_docs=args.max_docs))
        else:
            train_documents, val_documents = split_by_hash(
                documents, args.val_ratio, seed=args.seed,
            )
        train_tokens = tokenize_documents(
            train_documents, tokenizer, max_tokens=args.max_tokens,
        )
        val_tokens = tokenize_documents(
            val_documents, tokenizer, max_tokens=args.max_tokens,
        ) if val_documents else np.empty(0, dtype=np.int64)
        if args.save_tokens:
            np.save(Path(args.output_dir) / "train_tokens.npy", train_tokens)
            if val_tokens.size:
                np.save(Path(args.output_dir) / "val_tokens.npy", val_tokens)
        train_dataset = PackedTokenDataset(train_tokens, config.block_size)
        val_dataset = (
            PackedTokenDataset(val_tokens, config.block_size)
            if val_tokens.size >= config.block_size else None
        )
        manifest = {
            "mode": "pretrain", "data": _path_str(args.data),
            "train_tokens_file": _path_str(Path(args.output_dir) / "train_tokens.npy")
            if args.save_tokens else None,
            "val_tokens_file": _path_str(Path(args.output_dir) / "val_tokens.npy")
            if args.save_tokens and val_tokens.size else None,
            "hf_dataset": args.hf_dataset, "hf_config": args.hf_config,
            "hf_split": (args.hf_split or "train") if args.hf_dataset else None,
            "hf_text_field": args.hf_text_field if args.hf_dataset else None,
            "hf_streaming": args.hf_streaming if args.hf_dataset else None,
            "val_data": _path_str(args.val_data),
            "train_documents": len(train_documents), "val_documents": len(val_documents),
            "train_tokens": int(train_tokens.size), "val_tokens": int(val_tokens.size),
            "block_size": config.block_size, "vocab_size": config.vocab_size,
            "seed": args.seed, "val_ratio": args.val_ratio,
            "max_docs": args.max_docs, "max_tokens": args.max_tokens,
        }
        return train_dataset, val_dataset, manifest

    if args.hf_dataset:
        conversations = list(iter_hf_conversations(
            args.hf_dataset, config_name=args.hf_config, split=args.hf_split or "train",
            streaming=args.hf_streaming, max_docs=args.max_docs,
        ))
    else:
        conversations = list(iter_conversations(args.data, max_docs=args.max_docs))
    if not conversations:
        raise SystemExit("No conversations found in the training source")
    if args.val_data:
        train_conversations = conversations
        val_conversations = list(iter_conversations(args.val_data, max_docs=args.max_docs))
    else:
        train_conversations, val_conversations = split_by_hash(
            conversations, args.val_ratio, seed=args.seed,
        )
    train_dataset = SFTDataset(train_conversations, tokenizer, config.block_size)
    try:
        val_dataset = SFTDataset(val_conversations, tokenizer, config.block_size) if val_conversations else None
    except ValueError:
        val_dataset = None
    manifest = {
        "mode": "sft", "data": _path_str(args.data),
        "hf_dataset": args.hf_dataset, "hf_config": args.hf_config,
        "hf_split": (args.hf_split or "train") if args.hf_dataset else None,
        "hf_text_field": args.hf_text_field if args.hf_dataset else None,
        "hf_streaming": args.hf_streaming if args.hf_dataset else None,
        "val_data": _path_str(args.val_data),
        "train_conversations": len(train_conversations),
        "val_conversations": len(val_conversations),
        "train_examples": len(train_dataset),
        "block_size": config.block_size, "vocab_size": config.vocab_size,
        "seed": args.seed, "val_ratio": args.val_ratio,
        "max_docs": args.max_docs, "max_tokens": args.max_tokens,
    }
    return train_dataset, val_dataset, manifest


def _build_train_config(args: argparse.Namespace, saved: dict[str, Any] | None) -> TrainConfig:
    values: dict[str, Any] = {}
    for field in fields(TrainConfig):
        argument = getattr(args, field.name, None)
        if argument is not None:
            values[field.name] = argument
        elif saved and field.name in saved:
            values[field.name] = saved[field.name]
    values["output_dir"] = str(args.output_dir)
    return TrainConfig(**values)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resumed_config: ModelConfig | None = None
    saved_train: dict[str, Any] | None = None
    saved_manifest: dict[str, Any] = {}
    if args.resume:
        resumed_config, saved_train = _peek_checkpoint(args.resume)
        if not args.smoke_test:
            manifest_path = _resolve_checkpoint(args.resume).parent / "data_manifest.json"
            if manifest_path.exists():
                saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if args.data is None and saved_manifest.get("data"):
                    args.data = Path(saved_manifest["data"])
                if args.val_data is None and saved_manifest.get("val_data"):
                    args.val_data = Path(saved_manifest["val_data"])
                if args.hf_dataset is None and saved_manifest.get("hf_dataset"):
                    args.hf_dataset = saved_manifest["hf_dataset"]
                    args.hf_config = saved_manifest.get("hf_config")
                    args.hf_split = saved_manifest.get("hf_split")
                    args.hf_text_field = saved_manifest.get("hf_text_field")
                if args.mode is None and saved_manifest.get("mode") in {"pretrain", "sft"}:
                    args.mode = saved_manifest["mode"]
                cached_tokens = saved_manifest.get("train_tokens_file")
                if args.train_tokens is None and cached_tokens and Path(cached_tokens).exists():
                    args.train_tokens = Path(cached_tokens)
                    if saved_manifest.get("val_tokens_file"):
                        args.val_tokens = Path(saved_manifest["val_tokens_file"])
                    args.data = None
                    args.hf_dataset = None
                if args.max_docs is None:
                    args.max_docs = saved_manifest.get("max_docs")
                if args.max_tokens is None:
                    args.max_tokens = saved_manifest.get("max_tokens")
                if args.val_ratio is None and saved_manifest.get("val_ratio") is not None:
                    args.val_ratio = float(saved_manifest["val_ratio"])
                if args.seed is None and saved_manifest.get("seed") is not None:
                    args.seed = int(saved_manifest["seed"])
                if args.hf_streaming is None and saved_manifest.get("hf_streaming") is not None:
                    args.hf_streaming = bool(saved_manifest["hf_streaming"])

    if args.hf_streaming is None:
        args.hf_streaming = True
    if args.mode is None:
        args.mode = "pretrain"
    if args.seed is None:
        args.seed = 42
    if args.val_ratio is None:
        args.val_ratio = 0.01
    if args.output_dir is None:
        args.output_dir = _resolve_checkpoint(args.resume).parent if args.resume else Path("runs/lyra")
    if args.smoke_test:
        args.preset = "tiny"
        args.max_steps = args.max_steps or 20
        args.batch_size = args.batch_size or 4
        args.log_interval = args.log_interval or 5
        args.eval_interval = 10 if args.eval_interval is None else args.eval_interval
        args.save_interval = 20 if args.save_interval is None else args.save_interval
        if args.output_dir == Path("runs/lyra"):
            args.output_dir = Path("runs/smoke")

    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: LyraTokenizer | None = None
    if not args.smoke_test:
        if args.tokenizer_dir is None and args.resume:
            candidate = _resolve_checkpoint(args.resume).parent / "tokenizer"
            if candidate.exists():
                args.tokenizer_dir = candidate
        tokenizer = _load_tokenizer(args)
    config = _build_model_config(args, tokenizer, resumed_config)
    train_config = _build_train_config(args, saved_train)
    train_dataset, val_dataset, manifest = _build_datasets(args, tokenizer, config)

    config.save(args.output_dir / "model_config.json")
    train_config.save(args.output_dir / "train_config.json")
    (args.output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    if tokenizer is not None:
        tokenizer.save(args.output_dir / "tokenizer")

    model = LyraModel(config)
    summary = model.parameter_summary()
    print(
        f"Model parameters: {summary['total']:,} (trainable {summary['trainable']:,}); "
        f"train examples: {len(train_dataset)}; val examples: {len(val_dataset) if val_dataset else 0}",
        flush=True,
    )
    collator = CausalLMCollator(tokenizer.pad_token_id if tokenizer is not None else 0)
    trainer = Trainer(
        model, train_dataset, train_config, val_dataset=val_dataset,
        collator=collator, tokenizer=tokenizer,
    )
    print(
        f"Device: {trainer.device}; dtype: {trainer.dtype}; "
        f"steps: {train_config.max_steps}; batch: {train_config.batch_size}",
        flush=True,
    )
    history = trainer.train(resume_from=args.resume)
    final = trainer.evaluate()
    print(f"Final metrics: {json.dumps(final, ensure_ascii=False)}", flush=True)
    if args.smoke_test:
        if not history or not (args.output_dir / "final.pt").exists():
            raise SystemExit("SMOKE TEST FAIL")
        print("SMOKE TEST PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
