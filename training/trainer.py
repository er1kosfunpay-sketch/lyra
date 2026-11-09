"""Training loop with AMP, gradient accumulation, evaluation and resumable checkpoints."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from config.train_config import TrainConfig
from data.dataset import CausalLMCollator

CHECKPOINT_FORMAT = "lyra-checkpoint-v1"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            # GradScaler is CUDA-only; stay in FP32 unless the user forces FP16.
            return torch.float32
        return torch.float32
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = mapping[name]
    if dtype == torch.float16 and device.type == "cpu":
        raise ValueError("float16 autocast is not supported on CPU; use bfloat16 or float32")
    return dtype


class Trainer:
    def __init__(self, model, train_dataset: Dataset, config: TrainConfig, *,
                 val_dataset: Dataset | None = None, collator: Any | None = None,
                 tokenizer: Any | None = None) -> None:
        self.config = config
        self.raw_model = model
        self.model = model
        self.tokenizer = tokenizer
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.raw_model.to(self.device)
        if config.gradient_checkpointing:
            if not hasattr(self.raw_model, "enable_gradient_checkpointing"):
                raise TypeError("Model does not support gradient checkpointing")
            self.raw_model.enable_gradient_checkpointing(True)
        if config.compile:
            try:
                self.model = torch.compile(self.raw_model)
            except Exception as exc:  # pragma: no cover - depends on the local runtime
                print(f"torch.compile unavailable, continuing eagerly: {exc}")
                self.model = self.raw_model

        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0)) if tokenizer is not None else 0
        self.collator = collator or CausalLMCollator(pad_token_id)
        pin_memory = bool(config.pin_memory and self.device.type == "cuda")
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        self.train_generator = generator
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=self.collator, num_workers=config.num_workers,
            pin_memory=pin_memory, drop_last=False, generator=generator,
        )
        self.val_loader = None if val_dataset is None else DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=self.collator, num_workers=config.num_workers,
            pin_memory=pin_memory, drop_last=False,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty")

        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for _, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        groups = [{"params": decay, "weight_decay": config.weight_decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            groups, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.dtype == torch.float16 and self.device.type == "cuda"),
        )
        self.step = 0
        self.history: list[dict[str, Any]] = []
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _lr_lambda(self, step: int) -> float:
        warmup = self.config.warmup_steps
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if self.config.max_steps <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, self.config.max_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    @contextmanager
    def _autocast(self):
        enabled = self.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype if enabled else torch.float32,
            enabled=enabled,
        ):
            yield

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _record(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        log_path = Path(self.config.output_dir) / "metrics.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def train(self, *, resume_from: str | Path | None = None) -> list[dict[str, Any]]:
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        if self.step >= self.config.max_steps:
            print(f"Already at step {self.step} (max_steps={self.config.max_steps})")
            return self.history
        self.raw_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        accumulated_loss = 0.0
        accumulated_batches = 0
        tokens_since_log = 0
        log_start = time.perf_counter()
        train_start = time.perf_counter()
        try:
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    batch = self._move(batch)
                    with self._autocast():
                        outputs = self.model(
                            batch["input_ids"], attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels"),
                        )
                        loss = outputs["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.step + 1}: {float(loss)}"
                        )
                    self.scaler.scale(loss / self.config.grad_accum_steps).backward()
                    micro_step += 1
                    accumulated_loss += float(loss.detach())
                    accumulated_batches += 1
                    tokens_since_log += int(batch["input_ids"].numel())
                    if micro_step < self.config.grad_accum_steps:
                        continue

                    self.scaler.unscale_(self.optimizer)
                    max_norm = self.config.grad_clip if self.config.grad_clip > 0 else float("inf")
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), max_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at step {self.step + 1}: {float(grad_norm)}"
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.step += 1
                    micro_step = 0

                    if self.step % self.config.log_interval == 0 or self.step == 1:
                        now = time.perf_counter()
                        elapsed = max(now - log_start, 1e-9)
                        self._record({
                            "step": self.step,
                            "loss": accumulated_loss / max(accumulated_batches, 1),
                            "lr": self.scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                            "tokens_per_second": tokens_since_log / elapsed,
                            "elapsed_seconds": now - train_start,
                        })
                        accumulated_loss = 0.0
                        accumulated_batches = 0
                        tokens_since_log = 0
                        log_start = time.perf_counter()

                    if self.config.eval_interval and self.step % self.config.eval_interval == 0:
                        metrics = self.evaluate()
                        if metrics:
                            self._record({"step": self.step, "eval": metrics})

                    if self.config.save_interval and self.step % self.config.save_interval == 0:
                        path = self.save_checkpoint(tag=f"step_{self.step:07d}")
                        print(f"Saved checkpoint: {path}", flush=True)

                    if self.step >= self.config.max_steps:
                        break
                else:
                    continue
                break
        except KeyboardInterrupt:
            path = self.save_checkpoint(tag="interrupted")
            print(f"Interrupted; checkpoint saved to {path}", flush=True)
            raise
        self.save_checkpoint(tag="final")
        self.save_run_metadata()
        return self.history

    def evaluate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        was_training = self.raw_model.training
        self.raw_model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move(batch)
                with self._autocast():
                    outputs = self.model(
                        batch["input_ids"], attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )
                loss = outputs["loss"]
                if torch.isfinite(loss):
                    total_loss += float(loss)
                    batches += 1
                if batches >= self.config.eval_batches:
                    break
        self.raw_model.train(was_training)
        if batches == 0:
            return {"loss": float("nan"), "perplexity": float("nan")}
        average = total_loss / batches
        return {
            "loss": average,
            "perplexity": math.exp(average) if average < 709 else float("inf"),
        }

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "data_loader": self.train_generator.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def save_checkpoint(self, *, tag: str | None = None,
                        path: str | Path | None = None) -> Path:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = output / f"{tag or f'step_{self.step:07d}'}.pt"
        path = Path(path)
        state = {
            "format": CHECKPOINT_FORMAT,
            "step": self.step,
            "model_config": asdict(self.raw_model.config),
            "train_config": asdict(self.config),
            "model_state": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "history": self.history,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        latest = output / "latest.pt"
        if path.resolve() != latest.resolve():
            shutil.copyfile(path, latest)
        self._rotate_checkpoints()
        return path

    def _rotate_checkpoints(self) -> None:
        keep = self.config.checkpoint_history
        if keep <= 0:
            return
        checkpoints = sorted(
            Path(self.config.output_dir).glob("step_*.pt"),
            key=lambda item: item.stat().st_mtime,
        )
        for old in checkpoints[:-keep]:
            old.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> int:
        path = Path(path)
        if path.is_dir():
            path = path / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint format in {path}")
        saved_model_config = state.get("model_config")
        if saved_model_config != asdict(self.raw_model.config):
            raise ValueError("Checkpoint model configuration does not match the current model")
        self.raw_model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        self.step = int(state["step"])
        self.history = list(state.get("history", []))
        rng = state.get("rng")
        if rng:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if "data_loader" in rng:
                self.train_generator.set_state(rng["data_loader"])
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"Resumed from {path} at step {self.step}", flush=True)
        return self.step

    def save_run_metadata(self) -> None:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_config.json").write_text(
            json.dumps(asdict(self.raw_model.config), indent=2), encoding="utf-8",
        )
        (output / "train_config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8",
        )
        if self.tokenizer is not None and hasattr(self.tokenizer, "save"):
            self.tokenizer.save(output / "tokenizer")
