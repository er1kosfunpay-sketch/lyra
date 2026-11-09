"""Validated training hyperparameters for Lyra."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "runs/lyra"
    max_steps: int = 1_000
    batch_size: int = 4
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 500
    checkpoint_history: int = 2
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    gradient_checkpointing: bool = False
    compile: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if min(self.batch_size, self.grad_accum_steps, self.log_interval) < 1:
            raise ValueError("batch_size, grad_accum_steps and log_interval must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.eval_interval < 0 or self.save_interval < 0:
            raise ValueError("eval_interval and save_interval must be non-negative")
        if self.eval_interval and self.eval_batches < 1:
            raise ValueError("eval_batches must be positive when evaluation is enabled")
        if self.checkpoint_history < 0:
            raise ValueError("checkpoint_history must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
