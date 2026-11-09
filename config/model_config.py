"""Validated decoder-only Transformer configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    # ruGPT3 tokenizer (50,257 base IDs) plus four added Lyra role markers.
    vocab_size: int = 50_261
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    intermediate_size: int = 3_072
    block_size: int = 1_024
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02
    # Cross-entropy is computed in FP32 in sequence chunks to bound peak memory.
    loss_chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head for GQA")
        if self.n_embd // self.n_head % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.vocab_size < 264 or min(self.n_layer, self.n_head, self.n_kv_head, self.n_embd,
                                        self.intermediate_size, self.block_size) <= 0:
            raise ValueError("Model dimensions are outside the valid range")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("rope_theta, rms_norm_eps and init_std must be positive")
        if self.loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
