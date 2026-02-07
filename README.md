# BTC Price Direction Prediction

Система прогнозирования направления цены Bitcoin с использованием данных CoinGlass API и моделей машинного обучения (Logistic Regression).

**Два типа моделей:**
- **Base** — прогнозирует, вырастет ли цена BTC через N дней
- **Range** — прогнозирует, будет ли волатильность (диапазон high−low) выше скользящей средней

---

## Содержание

1. [Установка](#установка)
2. [config.json — Конфигурация экспериментов](#1-configjson--конфигурация-экспериментов)
3. [Models_builder_pipeline.py — Обучение моделей](#2-models_builder_pipelinepy--обучение-моделей)
4. [Predictor.py — Получение прогнозов](#3-predictorpy--получение-прогнозов)
5. [API — REST-сервер](#4-api--rest-сервер)
6. [Структура проекта](#5-структура-проекта)
7. [Соглашения по данным](#6-соглашения-по-данным)

---

## Установка

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

Создайте файл `dev.env` в корне проекта:
```
COINGLASS_API_KEY=your_key_here
```

> API-ключ можно получить на [open-api-v4.coinglass.com](https://open-api-v4.coinglass.com).

---

## 1. config.json — Конфигурация экспериментов

Файл содержит массив `runs` — список конфигураций, по которым обучаются модели. Каждая конфигурация = один эксперимент = одна обученная модель.

### Полная структура

```json
{
    "runs": [
        {
            "name": "base_model_1d",
            "N_DAYS": 1,
            "threshold": 0.5,
            "ma_window": 14,
            "range_feats": ["range_pct", "range_pct_ma14"],
            "base_feats": [
                "sp500__open__diff1__lag15",
                "futures_open_interest_aggregated_history__close__pct1",
                "gold__high__diff1",
                "..."
            ]
        }
    ]
}
```

### Описание параметров

| Параметр | Тип | Обязательный | Описание |
|----------|-----|:------------:|----------|
| `name` | string | да | Уникальное имя конфигурации. **Определяет тип модели** (см. ниже) |
| `N_DAYS` | int | да | Горизонт прогноза в днях (1, 3, 5, 7, 20...) |
| `threshold` | float | нет | Порог классификации (по умолчанию `0.5`). Вероятность выше порога = предсказание "рост" |
| `base_feats` | list | да | Список признаков для обучения модели |
| `ma_window` | int | нет | Окно скользящей средней для range-признаков (по умолчанию `14`) |
| `range_feats` | list | нет | Дополнительные признаки волатильности (для range-моделей) |

### Как `name` определяет тип модели

Пайплайн проверяет подстроку в поле `name`:

| Подстрока в `name` | Тип модели | Целевая переменная | Пример имени |
|---------------------|------------|---------------------|--------------|
| `base_model` | Base | `y_up_{N}d` | `base_model_1d` |
| `range_model` | Range | `y_range_up_range_pct_N{N}_ma{W}` | `range_model_7d` |

**Важно:** если в `name` нет ни `base_model`, ни `range_model` — модель не обучится.

### Формат именования признаков

Признаки следуют паттерну: `{источник}__{метрика}__{суффикс}`

```
futures_open_interest_aggregated_history__close__pct1
│                                        │      │
│                                        │      └── суффикс (pct1 = % изменение за 1 день)
│                                        └── метрика из API
└── источник данных (название эндпоинта)
```

**Доступные суффиксы:**
- `__diff1` — первая разность (значение − значение вчера)
- `__pct1` — процентное изменение за 1 день
- `__lag{N}` — лаг на N дней (для Gold и S&P500, чтобы учесть разницу в часовых поясах)

### Пример: конфигурация с двумя моделями

```json
{
    "runs": [
        {
            "name": "base_model_1d",
            "N_DAYS": 1,
            "threshold": 0.5,
            "base_feats": [
                "spot_price_history__close__pct1",
                "futures_open_interest_aggregated_history__close__pct1",
                "futures_funding_rate_history__open__pct1"
            ]
        },
        {
            "name": "range_model_3d",
            "N_DAYS": 3,
            "threshold": 0.52,
            "ma_window": 21,
            "range_feats": ["range_pct", "range_pct_ma21"],
            "base_feats": [
                "spot_price_history__close__pct1",
                "futures_open_interest_aggregated_history__close__diff1",
                "premium__diff1"
            ]
        }
    ]
}
```

При запуске пайплайна оба эксперимента выполнятся последовательно.

---

## 2. Models_builder_pipeline.py — Обучение моделей

### Запуск

```bash
python Models_builder_pipeline.py
```

Скрипт последовательно обучает модели для **каждой конфигурации** из `config.json`.

### Полный пайплайн по шагам

```
 1. Загрузка данных из CoinGlass API (27 источников)
 2. Объединение всех DataFrame по дате (outer join)
 3. Нормализация OHLCV-колонок (ensure_spot_prefix)
 4. Заполнение пропусков (forward fill)
 5. Feature Engineering (diff1, pct1, imbalance-признаки)
 6. Добавление лагов для Gold и S&P500 (1, 3, 5, 7, 10, 15 дней)
 7. Создание целевой переменной (y_up_{N}d)
 8. Фильтрация: оставляем только последние 1250 дней
 9. Удаление колонок с > 30% пропусков
10. Удаление строк с NaN
11. Обучение модели (base или range в зависимости от name)
12. Генерация графиков (ROC, метрики по порогам, confusion matrix)
```

### Источники данных (27 штук)

Все данные загружаются через `get_features_from_API.py`:

| Категория | Источники |
|-----------|-----------|
| Цена BTC | Spot OHLCV |
| Open Interest | OI history, OI aggregated, OI stablecoin, OI coin-margin |
| Funding Rate | FR history, FR OI-weighted, FR volume-weighted |
| Long/Short | Global ratio, Top accounts ratio, Top positions ratio |
| Ликвидации | Liquidation history, Liquidation aggregated |
| Торговля | Taker buy/sell volume, Taker buy/sell aggregated, Net position |
| Ордербук | Orderbook ask/bids, Orderbook aggregated |
| Индексы | CGDI index, Coinbase premium |
| On-chain | LTH supply, STH supply, Active addresses, Reserve risk |
| Внешние рынки | S&P 500 (yfinance), Gold (yfinance) |
| Маржинальные | Bitfinex margin long/short |

### Walk-Forward Cross-Validation

Используется `TimeSeriesSplit` — специальная кросс-валидация для временных рядов, которая **не допускает утечку данных из будущего**:

```
Fold 1:  [===TRAIN===] [=TEST=]
Fold 2:  [=====TRAIN=====] [=TEST=]
Fold 3:  [========TRAIN========] [=TEST=]
Fold 4:  [==========TRAIN==========] [=TEST=]
```

На каждом фолде:
1. Модель обучается на исторических данных (TRAIN)
2. Тестируется на будущих данных (TEST) — данные, которые модель **не видела**
3. Вычисляются метрики: AUC, Accuracy, Precision, Recall, F1

**Выбор лучшей модели:** из всех фолдов берётся модель с лучшим значением метрики (по умолчанию — F1).

### ML Pipeline (sklearn)

Каждый фолд обучает pipeline из 3 шагов:

```
SimpleImputer(strategy="mean")  →  StandardScaler()  →  LogisticRegression(max_iter=3000, class_weight="balanced")
│                                  │                     │
│ Заполняет NaN средним            │ Нормализация        │ Логистическая регрессия
│ значением признака               │ (mean=0, std=1)     │ с балансировкой классов
```

### Обучение Base модели

**Что предсказывает:** вырастет ли цена BTC через N дней.
- Целевая переменная: `y_up_{N}d` (1 = цена выросла, 0 = упала)
- Использует только `base_feats`

**Шаг 1.** Добавьте конфигурацию в `config.json`:
```json
{
    "name": "base_model_1d",
    "N_DAYS": 1,
    "threshold": 0.5,
    "base_feats": [
        "spot_price_history__close__pct1",
        "futures_open_interest_aggregated_history__close__pct1",
        "futures_funding_rate_history__open__pct1"
    ]
}
```

**Шаг 2.** Запустите:
```bash
python Models_builder_pipeline.py
```

**Результат:**
```
Models/base_model_1d/
├── model_base_base_model_1d.joblib     # обученная модель
└── metrics_base_base_model_1d.json     # метрики + список признаков

graphics/base_model_1d/
├── ROC_BASE_OOS.png                    # ROC-кривая
├── Metrics_vs_threshold_BASE_OOF___y_up_1d.png  # метрики по порогам
└── Confusion_Matrix_base_model_1d_thr0.50.png   # матрица ошибок
```

### Обучение Range модели

**Что предсказывает:** будет ли будущая волатильность (диапазон high−low) выше текущей скользящей средней.
- Целевая переменная: `y_range_up_range_pct_N{N}_ma{W}`
- Использует `base_feats` + `range_feats`
- Дополнительные признаки: `range_pct = (high - low) / close`, `range_pct_ma{W}` (скользящая средняя)

**Шаг 1.** Добавьте конфигурацию в `config.json`:
```json
{
    "name": "range_model_3d",
    "N_DAYS": 3,
    "threshold": 0.52,
    "ma_window": 21,
    "range_feats": ["range_pct", "range_pct_ma21"],
    "base_feats": [
        "spot_price_history__close__pct1",
        "futures_open_interest_aggregated_history__close__pct1"
    ]
}
```

**Важно:**
- `name` **должно** содержать `range_model`
- `ma_window` — определяет окно скользящей средней (должно совпадать с числом в `range_pct_ma{W}`)

**Шаг 2.** Запустите:
```bash
python Models_builder_pipeline.py
```

**Результат:**
```
Models/range_model_3d/
├── model_range_range_model_3d.joblib
└── metrics_range_range_model_3d.json

graphics/range_model_3d/
├── ROC_RANGE_OOS.png
├── Metrics_vs_threshold_RANGE_OOF___y_range_up_range_pct_N3_ma21.png
└── Confusion_Matrix_range_model_3d_thr0.52.png
```

### Сравнение типов моделей

| | Base модель | Range модель |
|---|---|---|
| **Вопрос** | Цена вырастет через N дней? | Волатильность будет выше средней? |
| **Целевая переменная** | `y_up_{N}d` | `y_range_up_range_pct_N{N}_ma{W}` |
| **Признаки** | `base_feats` | `base_feats` + `range_feats` |
| **Доп. параметры** | — | `ma_window`, `range_feats` |
| **Применение** | Прогноз направления (long/short) | Фильтрация по волатильности, опционные стратегии |

### Формат metrics JSON

После обучения сохраняется файл метрик:
```json
{
    "config_name": "base_model_1d",
    "model_path": "Models/base_model_1d/model_base_base_model_1d.joblib",
    "target": "y_up_1d",
    "features": ["feature1", "feature2", "..."],
    "n_features": 14,
    "thr": 0.5,
    "best_metric": "f1",
    "best_fold_idx": 1,
    "auc": 0.5714,
    "acc": 0.5561,
    "precision": 0.6,
    "recall": 0.6286,
    "f1": 0.614
}
```

---

## 3. Predictor.py — Получение прогнозов

Класс для получения предсказаний от обученных моделей. Подгружает модель, скачивает свежие данные с API, применяет тот же feature engineering что и при обучении.

### Инициализация

```python
from Predictor import Predictor

predictor = Predictor(
    config_name="base_model_1d",    # имя конфигурации из config.json
    config_path="config.json",      # путь к конфигу (по умолчанию)
    env_path="dev.env"              # путь к файлу с API-ключом (по умолчанию)
)
```

При инициализации:
1. Определяет тип модели (`base` или `range`) из `config_name`
2. Загружает конфигурацию из `config.json`
3. Загружает обученную модель из `Models/{config_name}/`
4. Загружает список признаков из файла метрик
5. Инициализирует `FeaturesGetter` для работы с API

### Публичные методы

#### `predict(n_dates=10) -> list[PredictionResult]`

Генерирует прогнозы для **последних N дат** из доступных данных.

```python
results = predictor.predict(n_dates=10)

for r in results:
    print(f"{r.date}: {'UP' if r.prediction == 1 else 'DOWN'} (p={r.probability:.3f})")
```

#### `predict_by_dates(dates: list[str]) -> list[PredictionResult]`

Генерирует прогнозы для **конкретных дат** (формат `"YYYY-MM-DD"`).

```python
results = predictor.predict_by_dates(["2025-01-20", "2025-01-21", "2025-01-22"])
```

Если какая-то дата отсутствует в данных — будет выведено предупреждение, но остальные даты обработаются.

#### `save_predictions(predictions=None, output_path=None) -> str`

Сохраняет прогнозы в JSON-файл.

```python
# Сохранить с генерацией прогнозов
path = predictor.save_predictions()

# Сохранить существующие прогнозы в конкретный файл
path = predictor.save_predictions(predictions=results, output_path="my_predictions.json")
```

**Формат выходного JSON:**
```json
{
    "config_name": "base_model_1d",
    "model_type": "base",
    "n_days_horizon": 1,
    "generated_at": "2025-01-25T12:00:00",
    "predictions": [
        {"date": "2025-01-20", "prediction": 1, "probability": 0.654},
        {"date": "2025-01-21", "prediction": 0, "probability": 0.412}
    ]
}
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    date: str           # дата в формате "YYYY-MM-DD"
    prediction: int     # 0 (падение) или 1 (рост)
    probability: float  # вероятность роста (0.0 — 1.0)
```

### Доступные конфигурации моделей

| Конфигурация | Тип | Горизонт |
|---|---|---|
| `base_model_1d` | Base | 1 день |
| `base_model_3d` | Base | 3 дня |
| `base_model_5d` | Base | 5 дней |
| `base_model_7d` | Base | 7 дней |
| `range_model_1d` | Range | 1 день |
| `range_model_3d` | Range | 3 дня |
| `range_model_5d` | Range | 5 дней |
| `range_model_7d` | Range | 7 дней |
| `range_model_20d` | Range | 20 дней |

> Модели находятся в папке `Models/`. Чтобы добавить новые — обучите их через `Models_builder_pipeline.py`.

---

## 4. API — REST-сервер

REST API на FastAPI для получения прогнозов по HTTP.

### Запуск

**Разработка** (с авто-перезагрузкой при изменении кода):
```bash
uvicorn api.main:app --reload --port 8000
```

**Production:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

После запуска:
- Swagger UI (интерактивная документация): `http://localhost:8000/docs`
- ReDoc (альтернативная документация): `http://localhost:8000/redoc`

### Эндпоинты

#### `POST /api/v1/predictions` — Получить прогнозы

Возвращает прогнозы выбранной модели для указанных дат.

**Request:**
```json
{
    "model_name": "base_model_1d",
    "dates": ["2025-01-20", "2025-01-21"]
}
```

- `model_name` — имя модели (из таблицы доступных конфигураций)
- `dates` — список дат в формате `YYYY-MM-DD` (от 1 до 100 дат)

**Response (200):**
```json
{
    "model_name": "base_model_1d",
    "model_type": "base",
    "horizon_days": 1,
    "requested_dates": ["2025-01-20", "2025-01-21"],
    "found_dates": ["2025-01-20"],
    "missing_dates": ["2025-01-21"],
    "predictions": [
        {
            "date": "2025-01-20",
            "prediction": 1,
            "probability": 0.654
        }
    ]
}
```

- `found_dates` — даты, для которых удалось сделать прогноз
- `missing_dates` — даты, которых нет в данных

**Ошибки:**
- `404` — модель не найдена
- `400` — неверный формат дат
- `500` — ошибка при загрузке модели или предсказании

---

#### `GET /api/v1/models` — Список моделей

Возвращает список всех доступных моделей с их метриками.

**Response (200):**
```json
{
    "available_models": [
        {
            "name": "base_model_1d",
            "model_type": "base",
            "horizon_days": 1,
            "feature_count": 14,
            "metrics": {
                "auc": 0.5714,
                "accuracy": 0.5561,
                "precision": 0.6,
                "recall": 0.6286,
                "f1": 0.614,
                "threshold": 0.5
            }
        },
        {
            "name": "range_model_3d",
            "model_type": "range",
            "horizon_days": 3,
            "feature_count": 16,
            "metrics": { "..." }
        }
    ]
}
```

---

#### `GET /api/v1/health` — Health check

**Response (200):**
```json
{
    "status": "healthy",
    "models_loaded": {
        "base_model_1d": true,
        "range_model_7d": true
    }
}
```

`models_loaded` показывает только модели, которые были загружены в кэш (после первого запроса).

---

### Примеры использования

**curl:**
```bash
# Прогнозы
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "base_model_1d", "dates": ["2025-01-20"]}'

# Список моделей
curl http://localhost:8000/api/v1/models

# Health check
curl http://localhost:8000/api/v1/health
```

**Python (requests):**
```python
import requests

# Прогнозы
response = requests.post(
    "http://localhost:8000/api/v1/predictions",
    json={
        "model_name": "base_model_1d",
        "dates": ["2025-01-20", "2025-01-21"]
    }
)
data = response.json()
for pred in data["predictions"]:
    print(f"{pred['date']}: {pred['prediction']} (p={pred['probability']:.3f})")

# Список моделей с метриками
models = requests.get("http://localhost:8000/api/v1/models").json()
for m in models["available_models"]:
    print(f"{m['name']}: AUC={m['metrics']['auc']:.4f}, F1={m['metrics']['f1']:.4f}")
```

---

## 5. Структура проекта

```
├── config.json                      # Конфигурация экспериментов
├── dev.env                          # API-ключ CoinGlass
├── Models_builder_pipeline.py       # Главный пайплайн обучения
├── Predictor.py                     # Класс для прогнозирования
├── get_features_from_API.py         # Сбор данных из 27 источников
├── graphics_builder.py              # ROC-кривые, графики метрик, confusion matrix
├── requirements.txt                 # Зависимости
│
├── FeaturesGetterModule/            # Работа с CoinGlass API
│   ├── FeaturesGetter.py            # Класс-обёртка над API
│   ├── features_endpoints.json      # Конфигурация 20 эндпоинтов
│   └── helpers/                     # Утилиты обработки ответов API
│
├── FeaturesEngineer/                # Feature Engineering
│   └── FeaturesEngineer.py          # ensure_spot_prefix, add_y_up_custom, add_engineered_features
│
├── CorrelationsAnalyzer/            # Статистический анализ признаков
│   └── CorrelationsAnalyzer.py      # corr_report (FDR), group_effect_report (Cohen's d)
│
├── ModelsTrainer/                   # Обучение моделей
│   ├── logistic_reg_model_train.py  # Walk-forward CV, hyperparameter tuning
│   ├── base_model_trainer.py        # Пайплайн обучения base-моделей
│   └── range_model_trainer.py       # Пайплайн обучения range-моделей
│
├── api/                             # REST API (FastAPI)
│   ├── main.py                      # Приложение FastAPI
│   ├── schemas.py                   # Pydantic-схемы запросов/ответов
│   └── routers/
│       └── predictions.py           # Эндпоинты предсказаний
│
├── Models/                          # Обученные модели и метрики
│   ├── base_model_1d/
│   │   ├── model_base_base_model_1d.joblib
│   │   └── metrics_base_base_model_1d.json
│   ├── range_model_7d/
│   │   └── ...
│   └── ...
│
├── graphics/                        # Автосгенерированные графики
│   ├── base_model_1d/
│   │   ├── ROC_BASE_OOS.png
│   │   ├── Metrics_vs_threshold_BASE_OOF___y_up_1d.png
│   │   └── Confusion_Matrix_base_model_1d_thr0.50.png
│   └── ...
│
├── LoggingSystem/                   # Логирование в файл + консоль
├── notebooks/                       # Jupyter notebooks для экспериментов
└── logs.log                         # Лог последнего запуска пайплайна
```

---

## 6. Соглашения по данным

- Все DataFrame содержат колонку `date` (datetime)
- Признаки: `{source}__{metric}` (например, `futures_open_interest_history__close`)
- Производные: `__diff1`, `__pct1`, `__lag{N}`
- Целевая переменная base: `y_up_{N}d` (1 если цена выросла через N дней)
- Целевая переменная range: `y_range_up_range_pct_N{N}_ma{W}` (1 если волатильность выше MA)
- Данные для обучения: последние **1250 дней** от максимальной даты
- Колонки с > 30% пропусков автоматически отбрасываются
