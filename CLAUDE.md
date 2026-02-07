# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin price direction prediction system using CoinGlass API data. Trains Logistic Regression models to predict if BTC price will be higher after N days (1d, 3d, 5d, 7d horizons). Two model types: **base** (price direction) and **range** (volatility above MA baseline). Serves predictions via FastAPI.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train all models (reads config.json, saves to Models/)
python Models_builder_pipeline.py

# Start prediction API (port 8000)
uvicorn api.main:app --reload

# Quick local prediction test
python Predictor.py

# API key required in dev.env:
# COINGLASS_API_KEY=your_key_here
```

## Project Structure

```
.
├── api/                          # FastAPI prediction server
│   ├── main.py                   # App entry point, CORS, lifespan
│   ├── schemas.py                # Pydantic request/response models
│   └── routers/predictions.py    # POST /api/v1/predictions, GET /api/v1/models, GET /api/v1/health
├── FeaturesGetterModule/         # Data fetching layer
│   ├── FeaturesGetter.py         # CoinGlass API client + yfinance (S&P500, Gold)
│   ├── features_endpoints.json   # 18 CoinGlass endpoint configs (paths, default params)
│   └── helpers/
│       ├── _coinglass_get_dataframe.py       # HTTP → DataFrame
│       ├── _coinglass_normalize_time_to_date.py  # Unix timestamps → YYYY-MM-DD
│       ├── _prefix_columns.py                # Adds source prefix to columns
│       └── _merge_features_by_date.py        # merge_by_date(): outer join + dedup
├── FeaturesEngineer/
│   └── FeaturesEngineer.py       # ensure_spot_prefix, add_y_up_custom, add_engineered_features
├── ModelsTrainer/
│   ├── logistic_reg_model_train.py  # walk_forward_logreg, tune_logreg_timecv, add_lags, add_range_target
│   ├── base_model_trainer.py     # base_model_train_pipeline() → saves model + metrics
│   └── range_model_trainer.py    # range_model_train_pipeline() → saves model + metrics
├── CorrelationsAnalyzer/         # Statistical feature analysis (corr, Cohen's d)
├── Models/                       # Trained model artifacts
│   ├── base_model_{1,3,5,7}d/   # model_base_*.joblib + metrics_base_*.json
│   └── range_model_{1,3,5,7}d/  # model_range_*.joblib + metrics_range_*.json
├── Models_builder_pipeline.py    # Training orchestrator: main_pipeline() per config
├── Predictor.py                  # Inference: loads model, fetches data, predicts
├── get_features_from_API.py      # Fetches 27 datasets (CoinGlass + yfinance), returns list[DataFrame]
├── graphics_builder.py           # ROC, metrics vs threshold, confusion matrix plots
├── config.json                   # Experiment configs (name, N_DAYS, base_feats, threshold)
├── dev.env                       # COINGLASS_API_KEY
└── graphics/                     # Saved plots per config_name
```

## Critical: Training vs Prediction Pipeline Order

Both pipelines MUST follow the same order. Reference implementation in `Models_builder_pipeline.main_pipeline()`:

```
1. get_features() → merge_by_date(how="outer")
2. ensure_spot_prefix()
3. ffill() on all feature columns          ← fills NaN gaps from outer join
4. add_engineered_features(horizon=N)      ← diff1, pct1, imbalances
5. add_lags(gold+sp500, lags=(1,3,5,7,10,15))
6. add_y_up_custom(horizon=N)              ← binary target
7. [training only] date filter, dropna, column cleanup
```

`Predictor._fetch_and_prepare_data()` mirrors steps 1-6 (no cleanup needed for inference).

## API Endpoints

```
POST /api/v1/predictions   — batch predict: {models: ["base_model_1d"], dates: ["2025-01-20"]}
GET  /api/v1/models        — list models with metrics (AUC, F1, etc.)
GET  /api/v1/health        — server status + loaded predictors
GET  /docs                 — Swagger UI
```

**Predictor caching**: `predictions.py` caches Predictor instances in `_predictor_cache` (one per model name). Each Predictor caches `_prepared_df` with 1-hour TTL.

## Model Artifacts

Each model saves two files in `Models/{config_name}/`:
- `model_{type}_{name}.joblib` — sklearn Pipeline (SimpleImputer → StandardScaler → LogisticRegression)
- `metrics_{type}_{name}.json` — features list, quality metrics (AUC, acc, precision, recall, F1), threshold

**Feature source of truth for prediction**: `metrics_*.json["features"]` (saved at training time). Fallbacks: `model.feature_names_in_` → `config.json["base_feats"]`.

## Data Sources (27 datasets in get_features_from_API.py)

| Category | Count | Source |
|----------|-------|--------|
| Futures (OI, funding, long/short, liquidation, orderbook, taker) | 17 | CoinGlass API (Binance) |
| On-chain (LTH supply, active addresses, STH supply, reserve risk) | 4 | CoinGlass API |
| Exchange (Bitfinex margin, Coinbase premium, CGDI index) | 3 | CoinGlass API |
| Spot BTC OHLCV | 1 | CoinGlass API |
| S&P 500 OHLCV | 1 | yfinance (^GSPC) |
| Gold Futures OHLCV | 1 | yfinance (GC=F) |

## Data Conventions

- All DataFrames have `date` column (datetime, YYYY-MM-DD)
- Feature columns use prefix pattern: `{source}__{metric}` (e.g., `futures_open_interest_history__close`)
- Derived features append suffix: `__diff1`, `__pct1`, `__lag{N}`
- Imbalance features: `feat__taker_imbalance_v2`, `feat__orderbook_imbalance_usd`, etc.
- Target column: `y_up_{N}d` (binary: 1 if price higher after N days)
- Range target: `y_range_up_range_pct_N{N}_ma{W}` (binary: future range > SMA baseline)
- Range features: `range_pct`, `range_pct_ma14`

## Key Patterns

- TimeSeriesSplit for walk-forward validation (no future leakage)
- Pipeline: SimpleImputer(mean) → StandardScaler → LogisticRegression(max_iter=3000, class_weight=balanced)
- Best model selected by metric (F1 by default) across CV folds
- Coverage filtering: features need >= 85% non-null values (training)
- External market lags: Gold + S&P500 columns get lags (1, 3, 5, 7, 10, 15 days)
- Graphics saved to `graphics/{config_name}/`
- Logging: `LoggingSystem` redirects stdout to `logs.log` during training
