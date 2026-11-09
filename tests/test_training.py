from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from core.model import LyraModel
from data.dataset import PackedTokenDataset
from training.trainer import Trainer


def tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=300, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       intermediate_size=64, block_size=16, dropout=0.0)


def tiny_train_config(tmp_path, **overrides) -> TrainConfig:
    values = dict(
        output_dir=str(tmp_path), max_steps=6, batch_size=2, grad_accum_steps=1,
        lr=1e-2, min_lr_ratio=0.1, warmup_steps=1, weight_decay=0.0,
        log_interval=1, eval_interval=3, eval_batches=2, save_interval=3,
        checkpoint_history=2, seed=0, device="cpu", dtype="float32",
        num_workers=0, pin_memory=False,
    )
    values.update(overrides)
    return TrainConfig(**values)


def make_dataset() -> PackedTokenDataset:
    tokens = np.tile(np.arange(1, 17), 4)
    return PackedTokenDataset(tokens, block_size=16)


def test_trainer_runs_and_checkpoint_resumes(tmp_path) -> None:
    config = tiny_train_config(tmp_path)
    dataset = make_dataset()
    model = LyraModel(tiny_config())
    trainer = Trainer(model, dataset, config, val_dataset=dataset)
    history = trainer.train()
    assert trainer.step == 6
    assert history and all(np.isfinite(record["loss"]) for record in history if "loss" in record)
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "model_config.json").exists()

    resumed_config = replace(config, max_steps=8)
    resumed_model = LyraModel(tiny_config())
    resumed = Trainer(resumed_model, dataset, resumed_config, val_dataset=dataset)
    assert resumed.load_checkpoint(tmp_path / "latest.pt") == 6
    resumed.train()
    assert resumed.step == 8
    assert (tmp_path / "final.pt").exists()


def test_small_model_overfits_a_single_batch(tmp_path) -> None:
    config = tiny_train_config(
        tmp_path, max_steps=40, batch_size=1, lr=1e-2, warmup_steps=0,
        eval_interval=0, save_interval=0, log_interval=100,
    )
    dataset = PackedTokenDataset(np.arange(0, 16), block_size=16)
    model = LyraModel(tiny_config())
    trainer = Trainer(model, dataset, config, val_dataset=dataset)
    before = trainer.evaluate()["loss"]
    trainer.train()
    after = trainer.evaluate()["loss"]
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before


def test_gradient_checkpointing_forward_backward() -> None:
    model = LyraModel(tiny_config())
    model.enable_gradient_checkpointing(True)
    ids = torch.randint(0, 300, (2, 8))
    result = model(ids, labels=ids)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
