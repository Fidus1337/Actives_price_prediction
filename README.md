# BTC Price Direction Prediction

Система прогнозирования направления цены Bitcoin с использованием данных CoinGlass API и моделей машинного обучения (Logistic Regression).

## Установка

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

Создайте файл `dev.env` с API ключом:
```
COINGLASS_API_KEY=your_key_here
```

---

## Запуск

### 1. Predictor.py — Получение прогнозов

Модуль для получения предсказаний от обученных моделей.

**Использование в коде:**
```python
from Predictor import Predictor

# Инициализация с указанием конфигурации модели
predictor = Predictor("base_model_1d")

# Прогноз на последние N дат
results = predictor.predict(n_dates=10)

# Прогноз на конкретные даты
results = predictor.predict_by_dates(["2025-01-20", "2025-01-21"])

# Сохранение результатов в JSON
predictor.save_predictions(results)
```

**Запуск напрямую:**
```bash
python Predictor.py
```

При запуске напрямую выполняется прогноз для конфигурации `range_model_1d` на указанные даты.

**Доступные конфигурации моделей:**
- `base_model_1d`, `base_model_3d`, `base_model_5d`, `base_model_7d` — базовые модели
- `range_model_1d`, `range_model_3d`, `range_model_5d`, `range_model_7d` — модели с учетом волатильности

---

### 2. Models_builder_pipeline.py — Обучение моделей

Пайплайн для обучения всех моделей согласно конфигурациям.

**Запуск:**
```bash
python Models_builder_pipeline.py
```

Скрипт:
1. Загружает все конфигурации из `config.json`
2. Для каждой конфигурации:
   - Получает данные через CoinGlass API
   - Выполняет feature engineering
   - Обучает модель (base или range)
   - Сохраняет модель в `Models/{config_name}/`
   - Строит графики метрик в `graphics/{config_name}/`
3. Логирует процесс в `logs.log`

---

### 3. config.json — Конфигурация экспериментов

Файл содержит массив конфигураций для обучения моделей.

**Структура:**
```json
{
    "runs": [
        {
            "name": "base_model_1d",      // Имя конфигурации
            "N_DAYS": 1,                   // Горизонт прогноза (дней)
            "base_feats": [...]            // Список признаков для модели
        },
        {
            "name": "range_model_1d",
            "N_DAYS": 1,
            "ma_window": 14,               // Окно скользящей средней (для range моделей)
            "range_feats": [...],          // Дополнительные признаки волатильности
            "base_feats": [...]
        }
    ]
}
```

**Параметры:**
| Параметр | Описание |
|----------|----------|
| `name` | Уникальное имя конфигурации |
| `N_DAYS` | Горизонт прогноза: 1, 3, 5 или 7 дней |
| `base_feats` | Список базовых признаков |
| `range_feats` | Признаки волатильности (опционально) |
| `ma_window` | Окно MA для range-признаков (по умолчанию 14) |

---

### Обучение Base модели

**Base модель** — использует только базовые рыночные признаки для прогноза.

**Шаг 1.** Добавьте конфигурацию в `config.json`:
```json
{
    "name": "base_model_1d",
    "N_DAYS": 1,
    "base_feats": [
        "spot_price_history__close__pct1",
        "spot_price_history__close__diff1",
        "futures_open_interest_aggregated_history__close__pct1",
        "futures_liquidation_aggregated_history__aggregated_short_liquidation_usd__diff1",
        "premium__diff1"
    ]
}
```

**Важно:** Имя конфигурации должно содержать `base_model` — это определяет тип обучения.

**Шаг 2.** Запустите обучение:
```bash
python Models_builder_pipeline.py
```

**Результат:**
- Модель: `Models/base_model_1d/model_base_base_model_1d.joblib`
- Метрики: `Models/base_model_1d/metrics.json`
- Графики: `graphics/base_model_1d/`

---

### Обучение Range модели

**Range модель** — дополнительно учитывает волатильность (диапазон high-low) для фильтрации сигналов в узких диапазонах.

**Шаг 1.** Добавьте конфигурацию в `config.json`:
```json
{
    "name": "range_model_1d",
    "N_DAYS": 1,
    "ma_window": 14,
    "range_feats": [
        "range_pct",
        "range_pct_ma14"
    ],
    "base_feats": [
        "spot_price_history__close__pct1",
        "spot_price_history__close__diff1",
        "futures_open_interest_aggregated_history__close__pct1",
        "premium__diff1"
    ]
}
```

**Важно:**
- Имя должно содержать `range_model`
- Обязательно указать `ma_window` и `range_feats`

**Шаг 2.** Запустите обучение:
```bash
python Models_builder_pipeline.py
```

**Результат:**
- Модель: `Models/range_model_1d/model_range_range_model_1d.joblib`
- Метрики и графики аналогично base модели

---

### Сравнение типов моделей

| Характеристика | Base модель | Range модель |
|----------------|-------------|--------------|
| Признаки | Только `base_feats` | `base_feats` + `range_feats` |
| Волатильность | Не учитывает | Учитывает через `range_pct` |
| Параметры | `N_DAYS`, `base_feats` | + `ma_window`, `range_feats` |
| Применение | Общий прогноз направления | Фильтрация в периоды высокой волатильности |

---

### 4. API сервер

Запуск REST API для получения прогнозов:

```bash
uvicorn api.main:app --reload --port 8000
```

Документация доступна по адресу: `http://localhost:8000/docs`

---

## Структура проекта

### Основные модули

| Модуль | Описание |
|--------|----------|
| **FeaturesGetterModule/** | Получение данных с CoinGlass API |
| **FeaturesEngineer/** | Инженерия признаков (diff, pct, lag) |
| **CorrelationsAnalyzer/** | Анализ корреляций и эффектов признаков |
| **ModelsTrainer/** | Обучение и валидация моделей |
| **LoggingSystem/** | Система логирования |
| **api/** | REST API на FastAPI |

### FeaturesGetterModule/
- `FeaturesGetter.py` — класс-обертка над CoinGlass API
- `features_endpoints.json` — конфигурация эндпоинтов
- `helpers/` — функции для обработки API-ответов

### FeaturesEngineer/
- `FeaturesEngineer.py` — класс для создания признаков:
  - `ensure_spot_prefix()` — нормализация имен OHLCV-колонок
  - `add_y_up_custom()` — создание бинарной целевой переменной
  - `add_engineered_features()` — добавление diff/pct/lag признаков

### CorrelationsAnalyzer/
- `CorrelationsAnalyzer.py` — статистический анализ:
  - `corr_report()` — корреляции с p-values и FDR коррекцией
  - `group_effect_report()` — Cohen's d между группами y=0/y=1

### ModelsTrainer/
- `logistic_reg_model_train.py` — утилиты для walk-forward CV
- `base_model_trainer.py` — обучение базовых моделей
- `range_model_trainer.py` — обучение range-моделей

### Вспомогательные файлы

| Файл | Описание |
|------|----------|
| `get_features_from_API.py` | Сбор всех признаков из API |
| `graphics_builder.py` | Построение ROC-кривых и графиков метрик |
| `Predictor.py` | Класс для прогнозирования |
| `Models_builder_pipeline.py` | Пайплайн обучения |

### Директории данных

| Папка | Содержимое |
|-------|------------|
| `Models/` | Сохраненные модели (.joblib) и метрики |
| `graphics/` | Графики ROC и метрик по порогам |
| `notebooks/` | Jupyter notebooks для экспериментов |

---

## Соглашения по данным

- Все DataFrame содержат колонку `date` (datetime)
- Признаки: `{source}__{metric}` (например, `futures_open_interest_history__close`)
- Производные: `__diff1`, `__pct1`, `__lag{N}`
- Целевая переменная: `y_up_{N}d` (1 если цена выросла через N дней)
