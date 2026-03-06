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
│   ├── main.py                   # App entry point, CORS, lifespan (caches dataset on startup)
│   ├── schemas.py                # Pydantic request/response models
│   └── routers/predictions.py    # 5 endpoints under /api prefix
├── FeaturesGetterModule/         # Data fetching layer
│   ├── FeaturesGetter.py         # CoinGlass API client + yfinance (S&P500, Gold)
│   ├── features_endpoints.json   # 18 CoinGlass endpoint configs (paths, default params)
│   └── helpers/
│       ├── _coinglass_get_dataframe.py       # HTTP → DataFrame
│       ├── _coinglass_normalize_time_to_date.py  # Unix timestamps → YYYY-MM-DD
│       ├── _prefix_columns.py                # Adds source prefix to columns
│       └── _merge_features_by_date.py        # merge_by_date(): outer join + dedup
├── FeaturesEngineer/
│   ├── FeaturesEngineer.py       # ensure_spot_prefix, add_y_up_custom, add_engineered_features
│   └── ta_features.py            # add_ta_features_selected(): 8 TA indicators per asset
├── ModelsTrainer/
│   ├── logistic_reg_model_train.py  # walk_forward_logreg, tune_logreg_timecv, add_range_target
│   ├── base_model_trainer.py     # base_model_train_pipeline() → saves model + metrics
│   └── range_model_trainer.py    # range_model_train_pipeline() → saves model + metrics
├── CorrelationsAnalyzer/         # Statistical feature analysis (corr, Cohen's d)
│   └── CorrelationsAnalyzer.py   # corr_report, corr_table_with_pvalues, group_effect_report
├── LoggingSystem/
│   └── LoggingSystem.py          # Tees stdout to terminal + logs.log (used in training)
├── Models/                       # Trained model artifacts (8 subdirs)
│   ├── base_model_{1,3,5,7}d/   # model_base_*.joblib + metrics_base_*.json
│   └── range_model_{1,3,5,7}d/  # model_range_*.joblib + metrics_range_*.json
├── notebooks/                    # Jupyter notebooks for experiments
├── shared_data_cache.py          # SharedBaseDataCache — single shared data pipeline
├── Models_builder_pipeline.py    # Training orchestrator: main_pipeline() per config
├── Predictor.py                  # Inference: loads model, fetches data, predicts
├── Dataset_builder_pipeline.py      # Fetches 27 datasets (CoinGlass + yfinance), returns list[DataFrame]
├── PlotsBuilder/Plots_Builder.py # ROC, metrics vs threshold, confusion matrix plots
├── new_targets.py                # Experimental targets (triple barrier, vol-scaled, return-threshold)
├── config.json                   # Experiment configs (name, N_DAYS, base_feats, threshold)
├── Logs/available_features.json  # Auto-generated: all ~330 feature names in current pipeline
├── dev.env                       # COINGLASS_API_KEY
└── graphics/                     # Saved plots per config_name
```

## Critical: Data Pipeline (SharedBaseDataCache)

All data preparation runs through `SharedBaseDataCache._fetch_base_data()` — a single shared pipeline used by both training and prediction. This ensures consistency.

### Shared base pipeline (runs once, cached):

```
1.  get_features() → 27 DataFrames
2.  merge_by_date(how="outer", dedupe="last")
3.  sort by date
4.  ensure_spot_prefix()
5.  ffill() on all feature columns
6.  Date filter: keep last _DATE_WINDOW_DAYS=1000 days
7.  Drop _SPARSE_COLUMNS (12 hardcoded columns with too many NaNs)
8.  Re-ffill + dropna()
9.  add_engineered_features()          ← __diff1, __pct1, 4 imbalance features
10. add_ta_features_selected(gold)     ← 8 TA indicators (adx, cci, rsi, roc, atr, bbw, obv, mfi)
11. add_ta_features_selected(sp500)    ← 8 TA indicators
12. add_ta_features_selected(spot)     ← 8 TA indicators (volume_col_override)
13. _trim_to_longest_continuous_segment()
14. dropna()
15. Save available_features.json
```

### Training pipeline (per config, in `main_pipeline()`):

```
1. shared_cache.get_base_df()          ← copy of shared data (steps 1-15)
2. add_y_up_custom(horizon=N_DAYS)     ← binary target
3. dropna(subset=[target]) + dropna()  ← removes last N rows + TA lookback
4. Train: base or range model (n_splits=4, best_metric="accuracy")
```

### Prediction pipeline (per model, in `Predictor._fetch_and_prepare_data()`):

```
1. shared_cache.get_base_df()          ← same shared data
2. add_y_up_custom(horizon=n_days)
3. [range only] add_range_target(use_pct=True, baseline_shift=1, ma_window=self.ma_window)
```

## API Endpoints

Prefix: `/api` (NOT `/api/v1`)

```
POST /api/predictions        — batch predict: {models: [...], dates: [...], refresh_dataset: bool}
GET  /api/models             — list models with cv_avg_* metrics
GET  /api/health             — server status + loaded predictor names
GET  /api/dataset-status     — dataset load status, last_refreshed_at, shape
POST /api/system/train_models — re-train models (optional JSON config body)
GET  /docs                   — Swagger UI
```

**Predictor caching**: `predictions.py` caches Predictor instances in `_predictor_cache` (one per model name). Each Predictor uses `SharedBaseDataCache` with 1-hour TTL.

## Model Artifacts

Each model saves two files in `Models/{config_name}/`:
- `model_{type}_{name}.joblib` — sklearn Pipeline (SimpleImputer → StandardScaler → LogisticRegression)
- `metrics_{type}_{name}.json` — features, quality metrics, threshold, cv averages

Metrics JSON fields:
- Config: `config_name`, `model_path`, `target`, `features`, `n_features`, `thr`, `best_metric`, `best_fold_idx`
- Best fold OOS: `auc`, `acc`, `precision`, `recall`, `f1`, `n_oos_samples`
- CV averages: `cv_avg_auc`, `cv_avg_acc`, `cv_avg_precision`, `cv_avg_recall`, `cv_avg_f1`

**Feature source of truth for prediction**: `metrics_*.json["features"]` (saved at training time). Fallbacks: `model.feature_names_in_` → `config.json["base_feats"]`.

## Data Sources (27 datasets in Dataset_builder_pipeline.py)

| Category | Count | Source |
|----------|-------|--------|
| Futures OI (history, aggregated, stablecoin, coin-margin) | 4 | CoinGlass API (Binance) |
| Futures Funding (history, OI-weight, vol-weight) | 3 | CoinGlass API |
| Futures Long/Short (global, top account, top position) | 3 | CoinGlass API |
| Futures Net Position v2 | 1 | CoinGlass API |
| Futures Liquidation (history, aggregated) | 2 | CoinGlass API |
| Futures Orderbook (ask/bids, aggregated) | 2 | CoinGlass API |
| Futures Taker Volume (v2, aggregated) | 2 | CoinGlass API |
| Exchange (Bitfinex margin, Coinbase premium, CGDI index) | 3 | CoinGlass API |
| On-chain (LTH supply, active addresses, STH supply, reserve risk) | 4 | CoinGlass API |
| Spot BTC OHLCV | 1 | CoinGlass API |
| S&P 500 OHLCV | 1 | yfinance (^GSPC) |
| Gold Futures OHLCV | 1 | yfinance (GC=F) |

## Data Conventions

- All DataFrames have `date` column (datetime, YYYY-MM-DD)
- Feature columns use prefix pattern: `{source}__{metric}` (e.g., `futures_open_interest_history__close`)
- Derived features: `__diff1`, `__pct1` suffixes
- TA features: `{prefix}__ta_{indicator}` (e.g., `gold__ta_adx`, `sp500__ta_rsi`, `spot_price_history__ta_bbw`)
- 8 TA indicators per asset: `ta_adx`, `ta_cci`, `ta_rsi`, `ta_roc`, `ta_atr`, `ta_bbw`, `ta_obv`, `ta_mfi`
- Imbalance features: `feat__taker_imbalance_v2`, `feat__orderbook_imbalance_usd`, etc.
- Target column: `y_up_{N}d` (binary: 1 if price higher after N days)
- Range target: `y_range_up_range_pct_N{N}_ma{W}` (binary: future range > SMA baseline)
- Range features: `range_pct`, `range_pct_ma{W}`

## Key Patterns

- TimeSeriesSplit for walk-forward validation (no future leakage)
- Pipeline: SimpleImputer(mean) → StandardScaler → LogisticRegression(max_iter=3000, class_weight=balanced)
- Best model selected by metric (`accuracy` by default) across CV folds, n_splits=4
- Sparse columns explicitly dropped via `_SPARSE_COLUMNS` list (12 columns)
- TA features (24 total) replace the old lag-based feature engineering
- Graphics saved to `graphics/{config_name}/`: ROC, metrics-vs-threshold, confusion matrix
- Logging: `LoggingSystem` redirects stdout to `logs.log` during training
- `available_features.json` auto-generated on each data fetch — ground truth for available features

## Config Structure (config.json)

Array of 8 experiment objects. Each has:
- `config_name`: e.g. `"base_model_1d"`, `"range_model_3d"`
- `N_DAYS`: prediction horizon (1, 3, 5, 7)
- `base_feats`: list of feature column names for this model
- `threshold`: probability threshold for binary classification
- `ma_window`: (range models only) SMA window for baseline, typically 7 or 14
