"""Terminal chat and one-shot inference for a trained Lyra checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config.model_config import ModelConfig
from core.model import LyraModel
from hf_tokenizer import DEFAULT_TOKENIZER_ID, DEFAULT_TOKENIZER_REVISION, LyraTokenizer
from training.trainer import resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a Lyra checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint file or run directory")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--system", default="Ты — Lyra, полезный русскоязычный ассистент.")
    parser.add_argument("--prompt", help="One-shot prompt instead of interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def load_checkpoint(path: Path, device_name: str = "auto", dtype_name: str = "auto",
                    tokenizer_dir: Path | None = None,
                    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
                    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION
                    ) -> tuple[LyraModel, LyraTokenizer, torch.device, torch.dtype]:
    path = path / "latest.pt" if path.is_dir() else path
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = ModelConfig(**state["model_config"])
    model = LyraModel(config)
    model.load_state_dict(state["model_state"])
    run_dir = path.parent

    saved_tokenizer = run_dir / "tokenizer"
    if tokenizer_dir is not None:
        tokenizer = LyraTokenizer.load(tokenizer_dir)
    elif saved_tokenizer.exists():
        tokenizer = LyraTokenizer.load(saved_tokenizer)
    else:
        tokenizer = LyraTokenizer.from_pretrained(
            tokenizer_id, revision=tokenizer_revision,
        )
    if tokenizer.vocab_size != config.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match model "
            f"vocab ({config.vocab_size})"
        )
    device = resolve_device(device_name)
    return model, tokenizer, device, resolve_dtype(dtype_name, device)


def format_prompt(history: list[dict[str, str]], system: str, user: str) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|system|>{system}<|turn_end|>")
    for turn in history:
        parts.append(f"<|user|>{turn['user']}<|turn_end|>")
        parts.append(f"<|assistant|>{turn['assistant']}<|turn_end|>")
    parts.append(f"<|user|>{user}<|turn_end|><|assistant|>")
    return "".join(parts)


def generate_reply(model: LyraModel, tokenizer: LyraTokenizer, history: list[dict[str, str]],
                   system: str, user: str, args: argparse.Namespace,
                   device: torch.device) -> str:
    # Drop the oldest turns until the prompt plus response fits in the context.
    working = list(history)
    while True:
        prompt_ids = tokenizer.encode(format_prompt(working, system, user))
        if len(prompt_ids) + args.max_new_tokens <= model.config.block_size or not working:
            break
        working.pop(0)
    if len(prompt_ids) + args.max_new_tokens > model.config.block_size:
        prompt_ids = prompt_ids[-(model.config.block_size - args.max_new_tokens):]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # SFT trains <|turn_end|> as the end of an assistant reply; </s> remains the
    # document separator learned during pretraining. Stop on either.
    stop_ids = [tokenizer.special_token_ids["<|turn_end|>"]]
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    output = model.generate(
        input_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=stop_ids,
    )
    new_tokens = output[0, len(prompt_ids):].tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")
    model, tokenizer, device, dtype = load_checkpoint(
        args.checkpoint, args.device, args.dtype, args.tokenizer_dir,
        args.tokenizer_id, args.tokenizer_revision,
    )
    if args.max_new_tokens >= model.config.block_size:
        raise SystemExit(
            f"--max-new-tokens must be smaller than the context length "
            f"({model.config.block_size})"
        )
    model.to(device=device, dtype=dtype)
    model.eval()
    history: list[dict[str, str]] = []

    if args.prompt:
        reply = generate_reply(model, tokenizer, history, args.system, args.prompt, args, device)
        print(reply)
        return 0

    print("Lyra chat. Commands: /exit, /reset, /system <text>")
    while True:
        try:
            user = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history.clear()
            print("История очищена.")
            continue
        if user.startswith("/system "):
            args.system = user[len("/system "):]
            print("Системный промпт обновлён.")
            continue
        reply = generate_reply(model, tokenizer, history, args.system, user, args, device)
        print(f"Lyra> {reply}")
        history.append({"user": user, "assistant": reply})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
