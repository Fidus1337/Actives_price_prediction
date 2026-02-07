# BTC Price Direction Prediction

A Bitcoin price direction prediction system using CoinGlass API data and machine learning models (Logistic Regression).

**Two model types:**
- **Base** — predicts whether BTC price will be higher after N days
- **Range** — predicts whether volatility (high-low range) will exceed its moving average

---

## Table of Contents

1. [Installation](#installation)
2. [config.json — Experiment Configuration](#1-configjson--experiment-configuration)
3. [Models_builder_pipeline.py — Model Training](#2-models_builder_pipelinepy--model-training)
4. [Predictor.py — Making Predictions](#3-predictorpy--making-predictions)
5. [API — REST Server](#4-api--rest-server)
6. [Project Structure](#5-project-structure)
7. [Data Conventions](#6-data-conventions)

---

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `dev.env` file in the project root:
```
COINGLASS_API_KEY=your_key_here
```

> You can get an API key at [open-api-v4.coinglass.com](https://open-api-v4.coinglass.com).

---

## 1. config.json — Experiment Configuration

The file contains a `runs` array — a list of configurations used to train models. Each configuration = one experiment = one trained model.

### Full Structure

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

### Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `name` | string | yes | Unique configuration name. **Determines the model type** (see below) |
| `N_DAYS` | int | yes | Prediction horizon in days (1, 3, 5, 7, 20...) |
| `threshold` | float | no | Classification threshold (default `0.5`). Probability above threshold = "up" prediction |
| `base_feats` | list | yes | List of features for model training |
| `ma_window` | int | no | Moving average window for range features (default `14`) |
| `range_feats` | list | no | Additional volatility features (for range models) |

### How `name` Determines Model Type

The pipeline checks for a substring in the `name` field:

| Substring in `name` | Model Type | Target Variable | Example Name |
|----------------------|------------|-----------------|--------------|
| `base_model` | Base | `y_up_{N}d` | `base_model_1d` |
| `range_model` | Range | `y_range_up_range_pct_N{N}_ma{W}` | `range_model_7d` |

**Important:** if `name` contains neither `base_model` nor `range_model`, the model will not be trained.

### Feature Naming Convention

Features follow the pattern: `{source}__{metric}__{suffix}`

```
futures_open_interest_aggregated_history__close__pct1
│                                        │      │
│                                        │      └── suffix (pct1 = % change over 1 day)
│                                        └── metric from API
└── data source (endpoint name)
```

**Available suffixes:**
- `__diff1` — first difference (value - yesterday's value)
- `__pct1` — percentage change over 1 day
- `__lag{N}` — lag by N days (for Gold and S&P500, to account for timezone differences)

### Example: Configuration with Two Models

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

When the pipeline is launched, both experiments will run sequentially.

---

## 2. Models_builder_pipeline.py — Model Training

### Running

```bash
python Models_builder_pipeline.py
```

The script sequentially trains models for **each configuration** from `config.json`.

### Full Pipeline Steps

```
 1. Fetch data from CoinGlass API (27 sources)
 2. Merge all DataFrames by date (outer join)
 3. Normalize OHLCV columns (ensure_spot_prefix)
 4. Fill missing values (forward fill)
 5. Feature Engineering (diff1, pct1, imbalance features)
 6. Add lags for Gold and S&P500 (1, 3, 5, 7, 10, 15 days)
 7. Create target variable (y_up_{N}d)
 8. Filter: keep only the last 1250 days
 9. Drop columns with > 30% missing values
10. Drop rows with NaN
11. Train model (base or range depending on name)
12. Generate plots (ROC, metrics vs threshold, confusion matrix)
```

### Data Sources (27 total)

All data is fetched via `get_features_from_API.py`:

| Category | Sources |
|----------|---------|
| BTC Price | Spot OHLCV |
| Open Interest | OI history, OI aggregated, OI stablecoin, OI coin-margin |
| Funding Rate | FR history, FR OI-weighted, FR volume-weighted |
| Long/Short | Global ratio, Top accounts ratio, Top positions ratio |
| Liquidations | Liquidation history, Liquidation aggregated |
| Trading | Taker buy/sell volume, Taker buy/sell aggregated, Net position |
| Order Book | Orderbook ask/bids, Orderbook aggregated |
| Indices | CGDI index, Coinbase premium |
| On-chain | LTH supply, STH supply, Active addresses, Reserve risk |
| External Markets | S&P 500 (yfinance), Gold (yfinance) |
| Margin | Bitfinex margin long/short |

### Walk-Forward Cross-Validation

Uses `TimeSeriesSplit` — a time-series-specific cross-validation that **prevents data leakage from the future**:

```
Fold 1:  [===TRAIN===] [=TEST=]
Fold 2:  [=====TRAIN=====] [=TEST=]
Fold 3:  [========TRAIN========] [=TEST=]
Fold 4:  [==========TRAIN==========] [=TEST=]
```

On each fold:
1. The model trains on historical data (TRAIN)
2. Tests on future data (TEST) — data the model **has never seen**
3. Metrics are computed: AUC, Accuracy, Precision, Recall, F1

**Best model selection:** from all folds, the model with the best metric value is chosen (F1 by default).

### ML Pipeline (sklearn)

Each fold trains a 3-step pipeline:

```
SimpleImputer(strategy="mean")  →  StandardScaler()  →  LogisticRegression(max_iter=3000, class_weight="balanced")
│                                  │                     │
│ Fills NaN with the               │ Normalization       │ Logistic Regression
│ feature mean                     │ (mean=0, std=1)     │ with class balancing
```

### Training a Base Model

**What it predicts:** whether BTC price will be higher after N days.
- Target variable: `y_up_{N}d` (1 = price went up, 0 = went down)
- Uses only `base_feats`

**Step 1.** Add a configuration to `config.json`:
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

**Step 2.** Run:
```bash
python Models_builder_pipeline.py
```

**Output:**
```
Models/base_model_1d/
├── model_base_base_model_1d.joblib     # trained model
└── metrics_base_base_model_1d.json     # metrics + feature list

graphics/base_model_1d/
├── ROC_BASE_OOS.png                    # ROC curve
├── Metrics_vs_threshold_BASE_OOF___y_up_1d.png  # metrics vs threshold
└── Confusion_Matrix_base_model_1d_thr0.50.png   # confusion matrix
```

### Training a Range Model

**What it predicts:** whether future volatility (high-low range) will exceed the current moving average.
- Target variable: `y_range_up_range_pct_N{N}_ma{W}`
- Uses `base_feats` + `range_feats`
- Additional features: `range_pct = (high - low) / close`, `range_pct_ma{W}` (moving average)

**Step 1.** Add a configuration to `config.json`:
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

**Important:**
- `name` **must** contain `range_model`
- `ma_window` — defines the moving average window (must match the number in `range_pct_ma{W}`)

**Step 2.** Run:
```bash
python Models_builder_pipeline.py
```

**Output:**
```
Models/range_model_3d/
├── model_range_range_model_3d.joblib
└── metrics_range_range_model_3d.json

graphics/range_model_3d/
├── ROC_RANGE_OOS.png
├── Metrics_vs_threshold_RANGE_OOF___y_range_up_range_pct_N3_ma21.png
└── Confusion_Matrix_range_model_3d_thr0.52.png
```

### Model Type Comparison

| | Base Model | Range Model |
|---|---|---|
| **Question** | Will price be higher in N days? | Will volatility exceed its average? |
| **Target Variable** | `y_up_{N}d` | `y_range_up_range_pct_N{N}_ma{W}` |
| **Features** | `base_feats` | `base_feats` + `range_feats` |
| **Extra Parameters** | — | `ma_window`, `range_feats` |
| **Use Case** | Directional prediction (long/short) | Volatility filtering, options strategies |

### Metrics JSON Format

After training, a metrics file is saved:
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

## 3. Predictor.py — Making Predictions

A class for getting predictions from trained models. It loads the model, fetches fresh data from the API, and applies the same feature engineering as during training.

### Initialization

```python
from Predictor import Predictor

predictor = Predictor(
    config_name="base_model_1d",    # configuration name from config.json
    config_path="config.json",      # path to config (default)
    env_path="dev.env"              # path to API key file (default)
)
```

During initialization:
1. Determines model type (`base` or `range`) from `config_name`
2. Loads configuration from `config.json`
3. Loads the trained model from `Models/{config_name}/`
4. Loads the feature list from the metrics file
5. Initializes `FeaturesGetter` for API access

### Public Methods

#### `predict(n_dates=10) -> list[PredictionResult]`

Generates predictions for the **last N dates** from available data.

```python
results = predictor.predict(n_dates=10)

for r in results:
    print(f"{r.date}: {'UP' if r.prediction == 1 else 'DOWN'} (p={r.probability:.3f})")
```

#### `predict_by_dates(dates: list[str]) -> list[PredictionResult]`

Generates predictions for **specific dates** (format `"YYYY-MM-DD"`).

```python
results = predictor.predict_by_dates(["2025-01-20", "2025-01-21", "2025-01-22"])
```

If a date is missing from the data, a warning will be printed, but the remaining dates will be processed.

#### `save_predictions(predictions=None, output_path=None) -> str`

Saves predictions to a JSON file.

```python
# Save with auto-generated predictions
path = predictor.save_predictions()

# Save existing predictions to a specific file
path = predictor.save_predictions(predictions=results, output_path="my_predictions.json")
```

**Output JSON format:**
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
    date: str           # date in "YYYY-MM-DD" format
    prediction: int     # 0 (down) or 1 (up)
    probability: float  # probability of price increase (0.0 — 1.0)
```

### Available Model Configurations

| Configuration | Type | Horizon |
|---|---|---|
| `base_model_1d` | Base | 1 day |
| `base_model_3d` | Base | 3 days |
| `base_model_5d` | Base | 5 days |
| `base_model_7d` | Base | 7 days |
| `range_model_1d` | Range | 1 day |
| `range_model_3d` | Range | 3 days |
| `range_model_5d` | Range | 5 days |
| `range_model_7d` | Range | 7 days |
| `range_model_20d` | Range | 20 days |

> Models are stored in the `Models/` folder. To add new ones, train them via `Models_builder_pipeline.py`.

---

## 4. API — REST Server

REST API built with FastAPI for getting predictions over HTTP.

### Running

**Development** (with auto-reload on code changes):
```bash
uvicorn api.main:app --reload --port 8000
```

**Production:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

After launch:
- Swagger UI (interactive docs): `http://localhost:8000/docs`
- ReDoc (alternative docs): `http://localhost:8000/redoc`

### Endpoints

#### `POST /api/v1/predictions` — Get Predictions

Returns predictions from the selected model for the specified dates.

**Request:**
```json
{
    "model_name": "base_model_1d",
    "dates": ["2025-01-20", "2025-01-21"]
}
```

- `model_name` — model name (from the available configurations table)
- `dates` — list of dates in `YYYY-MM-DD` format (1 to 100 dates)

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

- `found_dates` — dates for which predictions were generated
- `missing_dates` — dates not found in the data

**Errors:**
- `404` — model not found
- `400` — invalid date format
- `500` — error loading model or generating prediction

---

#### `GET /api/v1/models` — List Models

Returns a list of all available models with their metrics.

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

#### `GET /api/v1/health` — Health Check

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

`models_loaded` only shows models that have been loaded into the cache (after the first request).

---

### Usage Examples

**curl:**
```bash
# Predictions
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "base_model_1d", "dates": ["2025-01-20"]}'

# List models
curl http://localhost:8000/api/v1/models

# Health check
curl http://localhost:8000/api/v1/health
```

**Python (requests):**
```python
import requests

# Predictions
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

# List models with metrics
models = requests.get("http://localhost:8000/api/v1/models").json()
for m in models["available_models"]:
    print(f"{m['name']}: AUC={m['metrics']['auc']:.4f}, F1={m['metrics']['f1']:.4f}")
```

---

## 5. Project Structure

```
├── config.json                      # Experiment configuration
├── dev.env                          # CoinGlass API key
├── Models_builder_pipeline.py       # Main training pipeline
├── Predictor.py                     # Prediction class
├── get_features_from_API.py         # Data collection from 27 sources
├── graphics_builder.py              # ROC curves, metric plots, confusion matrices
├── requirements.txt                 # Dependencies
│
├── FeaturesGetterModule/            # CoinGlass API wrapper
│   ├── FeaturesGetter.py            # API wrapper class
│   ├── features_endpoints.json      # Configuration for 20 endpoints
│   └── helpers/                     # API response processing utilities
│
├── FeaturesEngineer/                # Feature Engineering
│   └── FeaturesEngineer.py          # ensure_spot_prefix, add_y_up_custom, add_engineered_features
│
├── CorrelationsAnalyzer/            # Statistical feature analysis
│   └── CorrelationsAnalyzer.py      # corr_report (FDR), group_effect_report (Cohen's d)
│
├── ModelsTrainer/                   # Model training
│   ├── logistic_reg_model_train.py  # Walk-forward CV, hyperparameter tuning
│   ├── base_model_trainer.py        # Base model training pipeline
│   └── range_model_trainer.py       # Range model training pipeline
│
├── api/                             # REST API (FastAPI)
│   ├── main.py                      # FastAPI application
│   ├── schemas.py                   # Pydantic request/response schemas
│   └── routers/
│       └── predictions.py           # Prediction endpoints
│
├── Models/                          # Trained models and metrics
│   ├── base_model_1d/
│   │   ├── model_base_base_model_1d.joblib
│   │   └── metrics_base_base_model_1d.json
│   ├── range_model_7d/
│   │   └── ...
│   └── ...
│
├── graphics/                        # Auto-generated plots
│   ├── base_model_1d/
│   │   ├── ROC_BASE_OOS.png
│   │   ├── Metrics_vs_threshold_BASE_OOF___y_up_1d.png
│   │   └── Confusion_Matrix_base_model_1d_thr0.50.png
│   └── ...
│
├── LoggingSystem/                   # Logging to file + console
├── notebooks/                       # Jupyter notebooks for experiments
└── logs.log                         # Log from the last pipeline run
```

---

## 6. Data Conventions

- All DataFrames contain a `date` column (datetime)
- Features: `{source}__{metric}` (e.g., `futures_open_interest_history__close`)
- Derived features: `__diff1`, `__pct1`, `__lag{N}`
- Base target variable: `y_up_{N}d` (1 if price went up after N days)
- Range target variable: `y_range_up_range_pct_N{N}_ma{W}` (1 if volatility exceeds MA)
- Training data: last **1250 days** from the maximum date
- Columns with > 30% missing values are automatically dropped
