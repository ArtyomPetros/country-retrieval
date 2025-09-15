# Country Detection — Interview Assignment (README.mb)

Привет! Это моё решение тестового задания про определение страны документа по изображению. Я сделал не классификатор, а **retrieval-подход**: учу эмбеддинги документов и на инференсе сравниваю их с **прототипами стран** (средними эмбеддингами). Получилось быстро, просто расширять, и хорошо переносится с паспортов на ID/права.

## Что внутри коротко

* Бэкенд: **MobileNetV3-Small** → голова BN+Dropout+Linear → **256-D эмбеддинг** + L2-норм.
* Обучение: **batch-hard triplet loss** с **PK-семплингом** (P стран × K примеров).
* Инференс: косинус до **прототипов** стран (усреднённых эмбеддингов).
* Скорость: на MacBook M2 CPU у меня **\~88 мс/изобр.** при 256×256 (требование <1 c выполнено).
* Масштабируется: чтобы добавить страну, просто добавляю её картинки и **пересчитываю прототипы** (модель не трогаю).

---
## 1) Требования и установка

Нужен Python 3.10+.

```bash
# войти в папку проекта
cd country-retrieval

# создать и активировать venv
python3 -m venv .venv && source .venv/bin/activate

# поставить зависимости
pip install -r requirements.txt
```

> Если будете запускать на Mac M1/M2: предупреждения про MPS/pin\_memory/Albumentations — нормальные, можно игнорировать.

---

## 2) Данные

Ожидаю структуру:

```
dataset/
  BEL/*.jpg
  BGR/*.png
  ...
```

Если у вас ZIP:

```bash
python tools/unzip_dataset.py --zip "/путь/к/вашему/dataset.zip" --out "./dataset"
```

Если уже есть папка `./dataset`, шаг распаковки не нужен. Главное, чтобы внутри были подпапки стран с картинками.

---

## 3) Обучение

Я обучал без скачивания предобученных весов (чтобы не споткнуться об SSL). Так тоже хорошо сходится на синтетике.

```bash
python train.py \
  --data-root ./dataset \
  --outdir ./artifacts \
  --epochs 10 \
  --batch-size 64 \
  --img-size 256 \
  --embed-dim 256 \
  --p-classes 8 \
  --k-samples 8 \
  --lr 1e-3 \
  --pretrained 0
```

Скрипт сам сделает train/val-сплит, будет логировать лосс и **Recall\@1** на валидации. Лучшие веса попадут в `./artifacts/best.pt`.

---

## 4) Экспорт прототипов стран

После обучения извлекаю эмбеддинги всего датасета и считаю средние по каждой стране:

```bash
python export_prototypes.py \
  --data-root ./dataset \
  --weights ./artifacts/best.pt \
  --outdir ./artifacts
```

Получаем:

* `./artifacts/prototypes.pt` — тензор \[C, 256] (C — страны),
* `./artifacts/labels.json` — маппер индекс → название страны.

---

## 5) Инференс (одна картинка)

На M2 процессоре мне быстрее на **CPU** (для таких маленьких моделей), плюс делаю 1–2 «прогревочных» прогона:

```bash
# возьмём любой файл из датасета
IMG=$(find ./dataset -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 1)

python infer.py \
  --device cpu \
  --image "$IMG" \
  --weights ./artifacts/best.pt \
  --prototypes ./artifacts/prototypes.pt \
  --labels ./artifacts/labels.json \
  --img-size 256 \
  --topk 5 \
  --warmup 2 \
  --print-prec 6
```

У меня это даёт что-то вроде:

```
Inference time (model+sim): ~88.0 ms on cpu
Top-K:
USA: 1.000000
KGZ: 0.999993
ESP: 0.999992
MDA: 0.999991
EST: 0.999989
```


---

## 6) Оценка метрик на валидации

```bash
python eval.py \
  --data-root ./dataset \
  --weights ./artifacts/best.pt \
  --prototypes ./artifacts/prototypes.pt \
  --labels ./artifacts/labels.json
```


---

## 7) Почему не классификатор и чем это удобно

* **Добавление стран без переобучения**: докинул картинки новой страны, пересчитал прототипы — готово.
* **Обобщение на разные документы**: мы не завязаны на MRZ; модель учит устойчивые визуальные признаки (цвет, шрифт, эмблемы, печати, верстка).
* **Скорость**: один forward до 256-мерного эмбеддинга + косинус с C прототипами → микросекунды на сравнение, десятки миллисекунд на модель.

---

## 8) Результаты по скорости

* MacBook Pro (M2): **\~88 мс/изобр.** при 256×256, `--device cpu`, `--warmup 2`.
* На MPS бывает чуть дольше из-за старта; маленькие модели часто выгоднее крутить на CPU.

---

## 9) TorchScript (опционально)

Если нужен максимально стабильный latency:

```bash
python - <<'PY'
import torch
from model import Embedder
ckpt = torch.load('./artifacts/best.pt', map_location='cpu')
m = Embedder(embed_dim=ckpt.get('args',{}).get('embed_dim',256), pretrained=False)
m.load_state_dict(ckpt['model']); m.eval()
ex = torch.randn(1,3,256,256)
ts = torch.jit.trace(m, ex)
ts.save('./artifacts/embedder_script.pt')
print('saved artifacts/embedder_script.pt')
PY
```