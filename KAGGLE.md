# Lyra на Kaggle

Порядок запуска: проверьте GPU, выполните офлайн smoke-тест, проверьте загрузку
данных коротким прогоном и только затем увеличивайте объём. Полный прогон зависит
от доступности Kaggle GPU, интернета и выбранного HF-датасета; локальная проверка
не может подтвердить эти внешние условия.

## 1. Проверка окружения

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
```

В Kaggle включите GPU в настройках Notebook и, если используете HF streaming,
включите Internet. Сначала проверьте среду:

```python
import torch
assert torch.cuda.is_available(), "В настройках Notebook выберите GPU"
print(torch.__version__, torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
```

Установите зависимости в отдельной ячейке и проверьте проект. В Kaggle обычно
уже установлен CUDA-сбор PyTorch; команда оставит его, если версия удовлетворяет
requirements.txt:

```bash
pip install -r requirements.txt
python train.py --smoke-test --output-dir /kaggle/working/smoke
```

Продолжайте, только если в выводе есть `SMOKE TEST PASS`. Smoke-тест офлайн и не
проверяет сеть, скачивание датасета или фактическое обучение на GPU.

## 2. Проверка данных коротким pretraining

Сначала проверьте доступ к датасету и пройдите несколько шагов. Ограничение
`--max-docs` также ограничивает число документов, предварительно собираемых для
детерминированного train/validation split:

```bash
python train.py \
  --hf-dataset HuggingFaceFW/fineweb-2 \
  --hf-config rus_Cyrl \
  --hf-text-field text \
  --max-docs 2000 \
  --max-tokens 500000 \
  --output-dir /kaggle/working/lyra-check \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-steps 10 \
  --warmup-steps 2 \
  --gradient-checkpointing \
  --eval-interval 5 \
  --save-interval 5 \
  --log-interval 1
```

Проверьте конечные train/eval loss и наличие `/kaggle/working/lyra-check/final.pt`.
Если загрузка датасета завершилась ошибкой, проверьте Kaggle Internet, имя/config
датасета и доступность нужной ревизии.

## 3. Увеличение pretraining

После успешной проверки данных для T4/P100 (16 GB) увеличивайте объём постепенно.
Начните с одного микро-батча и accumulation:

```bash
python train.py \
  --hf-dataset HuggingFaceFW/fineweb-2 \
  --hf-config rus_Cyrl \
  --hf-text-field text \
  --max-docs 20000 \
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

## 4. SFT

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

## 5. Checkpoint и resume

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

## 6. Что контролировать

- `metrics.jsonl`: loss должен быть конечным; резкие скачки или `NaN` — повод
  остановиться и проверить данные/learning rate.
- VRAM: уменьшайте `--batch-size`, увеличивайте `--grad-accum-steps`.
- Скорость: `tokens_per_second` в логах; при нехватке времени уменьшайте
  `--max-tokens`/`--max-docs` или используйте token cache.
- Качество: perplexity и loss не заменяют ручную проверку ответов; после SFT
  прогоните `chat.py` на фиксированном наборе prompt-категорий.
