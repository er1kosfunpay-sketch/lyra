# Lyra на Kaggle

Короткий чек-лист для GPU-сессии. Полный прогон начинайте только после локального
smoke-теста и проверки доступной VRAM.

## 1. Проверка окружения

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
```

```bash
pip install -r requirements.txt
python train.py --smoke-test --output-dir /kaggle/working/smoke
```

Smoke-тест офлайн и не требует данных; он проверяет optimizer, eval и checkpoint.

## 2. Первый короткий pretraining

Для T4/P100 (16 GB) начните с одного микро-батча и accumulation:

```bash
python train.py \
  --hf-dataset HuggingFaceFW/fineweb-2 \
  --hf-config rus_Cyrl \
  --hf-text-field text \
  --max-docs 100000 \
  --max-tokens 20000000 \
  --output-dir /kaggle/working/lyra \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --max-steps 1000 \
  --lr 3e-4 \
  --warmup-steps 100 \
  --gradient-checkpointing \
  --save-tokens \
  --eval-interval 100 \
  --save-interval 100 \
  --log-interval 10
```

`--dtype auto` выберет bf16, если GPU его поддерживает, иначе fp16. Для A100/L4
можно попробовать `--batch-size 4 --grad-accum-steps 4`.

Если streaming-датасет недоступен, загрузите `.txt`/`.jsonl` в Kaggle Dataset и
передайте `--data /kaggle/input/...`.

## 3. SFT

```bash
python train.py \
  --mode sft \
  --data /kaggle/input/lyra-sft/sft.jsonl \
  --output-dir /kaggle/working/lyra-sft \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --max-steps 1000 \
  --gradient-checkpointing
```

## 4. Checkpoint и resume

В `/kaggle/working/lyra` сохраняются:

- `latest.pt` — последний шаг;
- `final.pt` — финальный чекпойнт;
- `step_XXXXXXX.pt` — история (ротация по `--checkpoint-history`);
- `model_config.json`, `train_config.json`, `data_manifest.json`;
- `tokenizer/`;
- `metrics.jsonl`;
- `train_tokens.npy`/`val_tokens.npy` при `--save-tokens`.

Для продолжения в новой сессии примонтируйте предыдущий output как Kaggle Dataset
и запустите:

```bash
python train.py \
  --resume /kaggle/input/lyra-checkpoint \
  --max-steps 3000 \
  --output-dir /kaggle/working/lyra
```

Data/manifest/model config восстанавливаются из checkpoint, если пути доступны.
Если token cache найден, повторная токенизация не выполняется. В новой сессии
абсолютные пути `/kaggle/working/...` из прошлого запуска могут быть недоступны;
тогда явно передайте `--data` или `--train-tokens` (и при необходимости
`--val-tokens`).

## 5. Что контролировать

- `metrics.jsonl`: loss должен быть конечным; резкие скачки или `NaN` — повод
  остановиться и проверить данные/learning rate.
- VRAM: уменьшайте `--batch-size`, увеличивайте `--grad-accum-steps`.
- Скорость: `tokens_per_second` в логах; при нехватке времени уменьшайте
  `--max-tokens`/`--max-docs` или используйте token cache.
- Качество: perplexity и loss не заменяют ручную проверку ответов; после SFT
  прогоните `chat.py` на фиксированном наборе prompt-категорий.
