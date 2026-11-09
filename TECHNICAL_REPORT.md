# Lyra 1.0 — технический отчёт

Последнее обновление: 2026-11-09 (Asia/Yerevan)

Здесь записываются только фактически полученные результаты. `NOT RUN` означает,
что соответствующее измерение ещё не выполнялось.

## Текущий статус

| Проверка / показатель | Статус | Фактический результат |
|---|---|---|
| Аудит workspace | PASS | `Документы/Lyra` исходно пустой, без Git и исходников |
| HF tokenizer suite | PASS | Round-trip, specials, padding, save/load; vocab 50 261 |
| Model forward/backward (tiny) | PASS | 2 слоя, d=32: logits `[2,8,300]`, loss и градиенты конечны |
| Model forward/backward (default 142M) | PASS | batch 1, seq 8, CPU: logits `[1,8,50261]`, loss 11.2113, все градиенты конечны, 1.06 s |
| Causal attention | PASS | Изменение токенов с позиции 2 не изменило logits позиций 0–1 |
| GQA KV cache | PASS | Инкрементальная генерация с cache совпала с полным проходом |
| Tied embeddings | PASS | `lm_head.weight` и `token_embeddings.weight` имеют один `data_ptr` |
| Gradient checkpointing | PASS | Forward/backward с `use_reentrant=False`, градиенты конечны |
| Полный набор тестов | PASS | `python -m pytest tests core/tests -q --tb=short`: **30 passed, 0 failed, 34.01 s** (последний прогон; 24–34 s в зависимости от кэша); CPU PyTorch 2.14.0 |
| Data pipeline | PASS | Packing, right-padding, SFT-маскирование, deterministic split, локальные и HF-локальные источники |
| Overfit test | PASS | Tiny-конфиг, 1 блок, 40 шагов: eval loss **5.7586 → 0.3000** |
| Checkpoint / resume | PASS | Resume с шага 6 на 8; отдельно SFT-run resume с шага 3 на 4 с восстановлением data manifest |
| Training smoke | PASS | 20 шагов, tiny 106 816 параметров, CPU; final eval loss 6.2543, perplexity 520.22 на случайных токенах |
| Generation | PASS | Greedy с KV-cache совпал с полным проходом; EOS останавливает генерацию; sampling/top-k/top-p работают |
| CPU inference | PASS | `chat.py` загрузил tiny-checkpoint и сгенерировал токены; качество языка не оценивалось |
| Параметры default-модели | PASS | **142 428 672** параметра, измерено на созданном объекте; tied embeddings, vocab 50 261 |
| Полный pretraining / SFT | NOT RUN | Обучение на реальном корпусе не запускалось |
| Training/validation loss, perplexity на реальных данных | NOT RUN | Нет прогона на реальном корпусе |
| GPU VRAM / throughput | NOT RUN | Локально доступен только CPU |
| Полноценные веса Lyra | NOT RUN | Есть только tiny smoke/SFT-checkpoint-файлы; финальной модели нет |

## Реализовано

- Собственное PyTorch-core: GQA, RoPE, RMSNorm, SwiGLU, causal attention,
  residual connections, tied embeddings, KV-cache и gradient checkpointing.
- `core/generation.py`: greedy/temperature/top-k/top-p, repetition penalty,
  EOS-остановка и KV-cache.
- `data/dataset.py`: `.txt`/`.jsonl`/`.json` и HF streaming/локальные datasets;
  pretraining-паковка, SFT с маскированием prompt-токенов, collator с right-padding,
  детерминированный split по хэшу.
- `training/trainer.py`: AdamW с раздельными weight-decay группами, cosine schedule
  с warmup, gradient accumulation, AMP (bf16/fp16), clipping, validation,
  JSONL-логи, checkpoint/resume с RNG и DataLoader state, ротация чекпойнтов.
- `train.py`: CLI для pretraining, SFT, HF streaming, resume и офлайн smoke-теста;
  `--save-tokens` кэширует pretraining-поток в `.npy`, resume использует кэш
  автоматически.
- `chat.py`: one-shot и интерактивный чат с историей, системным промптом и
  ограничением контекста.
- `config/train_config.py`: валидируемая конфигурация обучения.
- `README.md` и `KAGGLE.md`: команды запуска, форматы данных, GPU-чек-лист и
  ограничения.
- Тесты: `tests/test_data.py`, `tests/test_generation.py`,
  `tests/test_training.py`, `tests/test_hf_tokenizer.py`, `core/tests/test_model.py`.

## Фактические замеры

- Default-конфигурация: 142 428 672 параметра; embeddings 38 600 448;
  12 блоков 103 827 456; final norm 768.
- FP32-веса default-модели: примерно 543 MiB (без optimizer/активаций).
- Full-size CPU forward+backward: 1.06 s для batch 1, seq 8; это не показатель
  GPU-производительности.
- Tiny smoke (106 816 параметров, CPU): около 5.7–8.8 тыс. токенов/с на коротких
  блоках; скорость зависит от размера модели и машины.
- Overfit tiny-модели: eval loss 5.7586 → 0.3000 за 40 шагов.

## Ограничения и риски

- Полный pretraining/SFT не выполнялся; качество русского языка не измерено.
  Loss/perplexity и smoke-тесты не заменяют ручную оценку ответов.
- 142M-модель с AdamW требует существенно больше памяти, чем сами веса:
  optimizer states, градиенты и активации. Для 8–16 GB GPU нужны bf16/fp16,
  gradient checkpointing и accumulation; фактический пик нужно измерить на GPU.
- HF streaming с `split_by_hash` материализует выбранные документы в памяти;
  для больших корпусов задавайте `--max-docs`/`--max-tokens`.
- Tokenizer зафиксирован на ревизии `a9307e696cd3c5b7f953ff4cb19d76a4d81821d5`;
  vocab 50 261. При смене tokenizer нужно пересчитать embeddings и переобучить модель.
- Лицензии и качество конкретных ревизий датасетов нужно проверять перед
  распространением весов/данных.
- Windows-консоль может отображать кириллицу некорректно при старой code page;
  для чата используйте UTF-8 терминал (`chcp 65001`).
- Предупреждение стороннего `pytest-asyncio` о настройке event-loop scope не
  влияет на результаты тестов.
