# Распознавание рукописного текста для исторических русских документов

Распознавание строк скорописи в дореформенной орфографии. Данные — записи Макаровского
волостного суда 1913–1914 годов и датасет Digital Peter.

- [docs/plan.md](docs/plan.md) — статус, запуск на сервере, оценки времени и что делаем по
  результатам;
- [docs/validation.md](docs/validation.md) — как устроена валидация: данные, разбиение,
  метрики;
- [docs/experiments/](docs/experiments/README.md) — журнал экспериментов и отчёты по ним;
- `docs/report.pdf` — то же одним файлом.

## Установка

```bash
uv sync
uv run pytest
```

Данные лежат в `data/` (структура описана в [docs/validation.md](docs/validation.md)). Если
они в другом месте, например в Colab, путь задаётся переменной `DIPLOMA_DATA`.

## Оценка модели

```bash
uv run diploma data                     # строки, страницы и алфавиты по частям
uv run diploma eval diploma.models.trocr:TrOCR -d makarov:test -d peter:test --name <прогон>
uv run diploma compare                  # таблица по всем прогонам в runs/
```

Модель для `eval` — это любой объект с методом `predict(images) -> list[str | Prediction]`.
Подробности — в [docs/validation.md](docs/validation.md).

## Обучение бейзлайна CRNN + CTC

Модель: первые три стадии ResNet34 (сжимают картинку по высоте, но почти не трогают ширину),
двухслойный BiLSTM и CTC. Так устроены лучшие решения на Digital Peter. Код —
`src/diploma/models/crnn.py` и `src/diploma/training.py`.

Учить нужно на GPU (Colab, Kaggle). На CPU эпоха на 48 строках идёт около трёх минут.

```bash
# 1. Digital Peter с нуля (ResNet инициализируется весами ImageNet)
uv run diploma train --name crnn-peter --train peter:train --val peter:val --epochs 60

# 2. Макаровка с нуля, для сравнения
uv run diploma train --name crnn-makarov --train makarov:train --val makarov:val --epochs 100

# 3. Дообучение модели с Peter на Макаровке
uv run diploma train --name crnn-peter-makarov --init checkpoints/crnn-peter/best.pt \
    --train makarov:train --val makarov:val --epochs 60 --lr 3e-4

# оценка на тесте
uv run diploma eval diploma.models.crnn:load -a checkpoint=checkpoints/crnn-peter/best.pt \
    -a device=cuda -d peter:test -d makarov:test --name crnn-peter
```

После каждой эпохи модель проверяется на `--val` тем же кодом, что и `diploma eval`.
Чекпойнт с лучшим CER на первом val-наборе сохраняется как `checkpoints/<имя>/best.pt`,
история обучения — в `history.jsonl`. Тест при обучении не используется.

`--train makarov:train*4` повторяет набор 4 раза за эпоху: так можно уравнять маленькую
Макаровку с Peter при совместном обучении. При `--init` алфавит дополняется новыми символами,
а веса для старых символов сохраняются.

## Дообучение Kansallisarkisto

Готовая модель Национального архива Финляндии (DINOv2-large + ruRoBERTa, 0.6 млрд параметров)
дообучается на наших строках. Код — `src/diploma/trocr_training.py`. Нужна видеокарта от 16 ГБ.

```bash
uv run diploma train-trocr --name trocr-makarov --train makarov:train --val makarov:val --epochs 20
uv run diploma eval diploma.models.trocr:TrOCR -a checkpoint=checkpoints/trocr-makarov/best \
    -a device=cuda -d makarov:test --name trocr-makarov
```

`--freeze-decoder-epochs N` — рецепт авторов: первые N эпох учится только энкодер.
`scripts/server_run.sh` запускает оба варианта вместе с CRNN; что ждать по времени и что
делать по результатам — в [docs/plan.md](docs/plan.md).
