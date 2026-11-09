# Lyra 1.0

Lyra — decoder-only Transformer на PyTorch, реализованный с нуля: GQA, RoPE,
RMSNorm, SwiGLU, causal attention, KV-cache и tied embeddings. Модель обучается
со случайных весов; pretrained-веса не используются.

## Что уже готово

- **Модель**: 12 слоёв, `d_model=768`, 12 Q-heads, 4 KV-heads, `d_ff=3072`,
  контекст 1024, словарь 50 261 (ruGPT3 tokenizer + 4 role markers).
  Фактически измеренный размер default-конфигурации: **142 428 672 параметра**
  (tied embeddings).
- **Данные**: локальные `.txt`/`.jsonl`/`.json`, HF streaming (`--hf-dataset`),
  pretraining-паковка в блоки, SFT-маскирование loss только на ответах ассистента,
  детерминированный split по хэшу.
- **Обучение**: AdamW с weight decay-группами, cosine schedule + warmup,
  gradient accumulation, AMP (bf16/fp16), gradient checkpointing, clipping,
  validation loss/perplexity, JSONL-логи, checkpoint/resume с RNG и DataLoader state.
- **Инференс**: генерация с KV-cache, greedy/temperature/top-k/top-p,
  repetition penalty, остановка по EOS; терминальный чат.
- **Проверки**: 30 тестов, включая forward/backward, causal mask, KV-cache,
  overfit, checkpoint/resume, SFT-маскирование и генерацию.

Полный pretraining/SFT на реальном корпусе ещё **не запускался**, поэтому
обученных весов и оценки качества языка пока нет. Это следующий этап.

## Установка

```powershell
pip install -r requirements.txt
```

Требуется Python 3.12+ и PyTorch 2.3+. GPU не обязателен для smoke-тестов, но
нужен для реального обучения. Для Kaggle см. `KAGGLE.md`.

## Быстрый smoke-тест

Полностью офлайн, без скачивания данных и tokenizer:

```powershell
python train.py --smoke-test --output-dir runs/smoke
```

Проверяются forward/backward, optimizer, scheduler, eval, checkpoint и
сохранение `final.pt`. В конце печатается `SMOKE TEST PASS`.

## Pretraining на локальном тексте

```powershell
python train.py `
  --data data/train.txt `
  --val-ratio 0.01 `
  --max-tokens 20000000 `
  --output-dir runs/lyra-base `
  --batch-size 4 `
  --grad-accum-steps 8 `
  --max-steps 5000 `
  --lr 3e-4 `
  --warmup-steps 200 `
  --eval-interval 100 `
  --save-interval 500
```

Для 142M-модели на 8–16 GB GPU начните с `--batch-size 1`,
`--grad-accum-steps 16`, `--gradient-checkpointing`, `--dtype bfloat16`.
`--max-tokens` ограничивает объём токенов и полезен для первого прогона.

Добавьте `--save-tokens`, чтобы сохранить `train_tokens.npy`/`val_tokens.npy`
в output-dir. При последующем `--resume` эти файлы используются автоматически
вместо повторной токенизации. Их также можно передать явно через
`--train-tokens`/`--val-tokens`.

### HF streaming

```powershell
python train.py `
  --hf-dataset HuggingFaceFW/fineweb-2 `
  --hf-config rus_Cyrl `
  --hf-text-field text `
  --max-docs 200000 `
  --max-tokens 50000000 `
  --output-dir runs/lyra-base
```

Для больших корпусов обязательно задавайте `--max-docs`/`--max-tokens`:
split по хэшу материализует выбранные документы в памяти. Лицензии и качество
конкретной ревизии датасета нужно проверять отдельно.

## SFT

Формат JSONL (поддерживаются также `prompt`/`response`, `question`/`answer`,
Alpaca-style `instruction`/`input`/`output`):

```json
{"messages":[{"role":"system","content":"Ты полезный ассистент."},{"role":"user","content":"Привет!"},{"role":"assistant","content":"Здравствуйте! Чем помочь?"}]}
```

```powershell
python train.py `
  --mode sft `
  --data data/sft.jsonl `
  --val-ratio 0.02 `
  --output-dir runs/lyra-sft `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --max-steps 1000
```

Loss считается только на токенах ответа ассистента и `<|turn_end|>`; системные
и пользовательские реплики маскируются (`-100`).

## Resume

```powershell
python train.py --resume runs/lyra-base --max-steps 8000
```

Если `--data`/`--val-data`/`--mode` не переданы, они восстанавливаются из
`data_manifest.json`; `--model-config` — из checkpoint. `latest.pt` всегда
указывает на последний сохранённый шаг. При переносе между GPU/CPU явно задайте
`--device` и `--dtype`.

## Чат

```powershell
python chat.py runs/lyra-sft --prompt "Привет! Как дела?"
python chat.py runs/lyra-sft
```

Интерактивные команды: `/exit`, `/reset`, `/system <текст>`.

## Тесты

```powershell
pip install -r requirements-dev.txt
python -m pytest tests core/tests -q
```

## Структура

```text
config/            ModelConfig, TrainConfig
core/              attention, RoPE, RMSNorm, SwiGLU, Transformer, model, generation
data/              загрузка источников, tokenization, packing, SFT, collator
training/          Trainer, checkpoint/resume, eval
tests/             data/generation/training/tokenizer tests
train.py           CLI обучения
chat.py            CLI инференса
KAGGLE.md          чек-лист GPU-сессии
TECHNICAL_REPORT.md фактические результаты проверок
DATASET_REPORT.md   аудит источников данных
```

## Ограничения

- Полный pretraining на целевом объёме токенов ещё не выполнялся; качество
  русского языка не измерено.
- FP32-веса 142M-модели занимают примерно 543 MiB, optimizer states AdamW
  увеличивают training VRAM; фактический пик нужно измерять на целевой GPU.
- Tokenizer зафиксирован на ревизии `a9307e696cd3c5b7f953ff4cb19d76a4d81821d5`;
  при смене tokenizer нужно пересчитать `vocab_size` и переобучить модель.
- Метрики loss/perplexity не заменяют ручную оценку качества ответов.
- Checkpoint-файлы загружаются с `weights_only=False`, чтобы восстановить RNG
  и DataLoader state. Загружайте только собственные/доверенные чекпойнты.
