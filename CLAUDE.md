# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin price direction prediction system with two independent prediction pipelines served over one FastAPI service:

1. **Classic ML** — Logistic Regression on CoinGlass + yfinance features. Horizons 1d/3d/5d/7d, two model types: **base** (price direction) and **range** (volatility above MA baseline).
2. **Multiagent system** — LangGraph DAG of LLM-powered agents (Twitter analyser, Tech indicators analyser, Verdicts validator, Reports analyser) that vote on LONG/SHORT direction for a requested `forecast_start_date`.

Market data comes from **CoinGlass (Bybit)** for futures/on-chain and yfinance for S&P 500 / Gold / IGV.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train all classic ML models (reads configs/config.json → saves to Classic_ml_model_solutions/Created_models_to_use/)
python -m Classic_ml_model_solutions.Models_builder_pipeline.Models_builder_pipeline

# Start API (port 8000) — serves both classic ML and multiagent endpoints
uvicorn api.main:app --reload

# Run multiagent system standalone (reads configs/multiagent_config.json)
python -m MultiagentSystem.multiagent_system_main

# Tune multiagent hyperparameters (reads tuning_top.json grid)
python -m MultiagentSystem.hyperparameters_tuner

# Required env in dev.env:
# COINGLASS_API_KEY=...
# OPENAI_API_KEY=... (for multiagent LLM calls)
# TWITTER_UPLOAD_KEY=... (optional, for /api/agents/twitter-upload-cookies)
```

## Project Structure

```
.
├── api/                                    # FastAPI service (serves BOTH pipelines)
│   ├── main.py                             # App entry, CORS, lifespan — creates SharedBaseDataCache and wires routers
│   ├── schemas.py                          # Pydantic request/response models for both routers
│   └── routers/
│       ├── classic_ml_predictions.py       # Classic ML: /api/predictions, /api/models, /api/health, /api/dataset-status, /api/system/train_classic_ml_models
│       └── multiagent_predictions.py       # Multiagent: /api/multiagent_predictions, /api/system/collect_agent_data, /api/agents/*
├── Classic_ml_model_solutions/Dataset_pipeline/
│   ├── FeaturesGetterModule/               # Data fetching layer (CoinGlass + yfinance)
│   │   ├── FeaturesGetter.py               # CoinGlass client + yfinance wrappers (S&P500, Gold, IGV)
│   │   ├── features_endpoints.json         # 18 CoinGlass endpoint configs — default exchange pinned to Bybit
│   │   └── helpers/                        # _coinglass_get_dataframe / _normalize_time_to_date / _prefix_columns / _merge_features_by_date
│   ├── FeaturesEngineer/
│   │   ├── FeaturesEngineer.py             # ensure_spot_prefix, add_y_up_custom, add_engineered_features, add_price_ma_features
│   │   └── ta_features.py                  # add_ta_features_selected(): 8 TA indicators per asset
├── Classic_ml_model_solutions/Filtering_features_pipeline/
│   └── CorrelationsAnalyzer/CorrelationsAnalyzer.py    # corr_report, corr_table_with_pvalues, group_effect_report
├── Classic_ml_model_solutions/Models_builder_pipeline/ModelsTrainer/
│   ├── logistic_reg_model_train.py         # walk_forward_logreg, tune_logreg_timecv, add_range_target
│   ├── base_model_trainer.py               # base_model_train_pipeline()
│   ├── range_model_trainer.py              # range_model_train_pipeline()
│   ├── ret_threshold_model_trainer.py      # ret_threshold_model_train_pipeline()
│   └── vol_scaled_model_trainer.py         # vol_scaled_model_train_pipeline()
├── Logs/LoggingSystem/LoggingSystem.py     # Tees stdout → logs.log (used during training/tuning)
├── Classic_ml_model_solutions/Created_models_to_use/  # classic-ML artifact dirs: base_model_{1,3,5,7}d/, range_model_{1,3,5,7}d/, ret_threshold_model_*d/, vol_scaled_model_*d/
├── Classic_ml_model_solutions/Dataset_pipeline/SharedDataCache/
│   ├── SharedBaseDataCache.py              # Single shared base-data pipeline with TTL + thread lock (used by classic ML and multiagent)
│   └── __init__.py
├── Classic_ml_model_solutions/Predict_with_ml_model/Predictor.py  # Classic-ML inference: loads joblib, reuses SharedBaseDataCache, returns probs
├── MultiagentSystem/                       # LangGraph agent DAG (see "Multiagent System" section below)
├── Classic_ml_model_solutions/Dataset_pipeline/Dataset_builder_pipeline.py  # get_features() — fetches 28 datasets in parallel (ThreadPoolExecutor)
├── Classic_ml_model_solutions/Models_builder_pipeline/Models_builder_pipeline.py  # Training orchestrator: main_pipeline() per config in configs/config.json
├── Classic_ml_model_solutions/PlotsBuilder/Plots_Builder.py  # ROC, metrics-vs-threshold, confusion matrix plots
├── new_targets.py                          # Experimental targets (triple barrier, vol-scaled, return-threshold)
├── configs/
│   ├── config.json                         # 8 classic-ML experiment configs
│   └── ml_config.json                      # Alternate feature sets (experimental)
├── notebooks/                              # Jupyter experiments
├── Logs/available_features.json            # Auto-generated on every data fetch — ground truth for feature names
├── dev.env                                 # COINGLASS_API_KEY, OPENAI_API_KEY, TWITTER_UPLOAD_KEY
└── graphics/                               # Saved plots per config_name
```

## Critical: Data Pipeline (SharedBaseDataCache)

All data preparation runs through `Classic_ml_model_solutions.Dataset_pipeline.SharedDataCache.SharedBaseDataCache._fetch_base_data()` — a single shared pipeline used by **classic ML training, classic ML prediction, and the multiagent system**. This ensures feature parity across all three. The cache is thread-safe (`threading.Lock`) with a TTL of 3600s; `api/main.py` creates one instance at startup, calls `.refresh()`, and injects it via `Predictor.set_shared_cache(...)`.

### Shared base pipeline (runs once, cached):

```
1.  get_features() → 28 DataFrames  (ThreadPoolExecutor, max_workers=8)
2.  merge_by_date(how="outer", dedupe="last") + sort by date
3.  ensure_spot_prefix()
4.  ffill() on all feature columns
5.  Date filter: keep last _DATE_WINDOW_DAYS=1000 days
6.  Drop _SPARSE_COLUMNS (12 columns — orderbook USD sides + cgdi index)
7.  Re-ffill + dropna()
8.  add_engineered_features()                 ← __diff1, __pct1, imbalance feats
9.  add_price_ma_features()                   ← SMA 7/14/21/50, __smaN_rel, __zscoreN
10. add_ta_features_selected(gold / sp500 / igv / spot_price_history)
                                              ← 8 TA indicators × 4 assets (adx, cci, rsi, roc, atr, bbw, obv, mfi)
11. Lag features: shift(1, 3, 5, 7, 15) for every non-diff/non-pct column
                                              ← _LAG_PERIODS=[1, 3, 5, 7, 15]
12. dropna() → _trim_to_longest_continuous_segment()
                                              ← order matters: dropna first, THEN trim
13. Write Logs/available_features.json
```

### Classic-ML training (per config, in `main_pipeline()`):

```
1. shared_cache.get_base_df()          ← copy of shared data
2. add_y_up_custom(horizon=N_DAYS)     ← binary target y_up_Nd
3. dropna(subset=[target]) + dropna()  ← removes last N rows + lookback tail
4. Train base or range model (TimeSeriesSplit n_splits=4, best_metric="accuracy")
```

### Classic-ML prediction (`Classic_ml_model_solutions/Predict_with_ml_model/Predictor._fetch_and_prepare_data()`):

```
1. shared_cache.get_base_df()
2. add_y_up_custom(horizon=n_days)
3. [range only] add_range_target(use_pct=True, baseline_shift=1, ma_window=self.ma_window)
```

### Multiagent prediction (`MultiagentSystem.multiagent_predictions_module.make_one_prediction`):

```
1. shared_cache.get_base_df()          ← same cached base df
2. Pass as state["cached_dataset"] into LangGraph app.invoke(...)
3. Each agent reads its own feature slice from the cached df via state
```

## API Endpoints

Prefix: `/api` (NOT `/api/v1`). All endpoints grouped by router:

### Classic ML (`api/routers/classic_ml_predictions.py`)
```
POST /api/predictions                     — batch predict: {models, dates, refresh_dataset}
GET  /api/models                          — list models with cv_avg_* metrics
GET  /api/health                          — server status + loaded predictor names
GET  /api/dataset-status                  — dataset load status, last_refreshed_at, shape
POST /api/system/train_classic_ml_models  — retrain classic ML models from configs/config.json
```

### Multiagent (`api/routers/multiagent_predictions.py`)
```
POST /api/multiagent_predictions          — run LangGraph system for N last eligible dates
                                            (body shaped like multiagent_config.json + n_last_dates)
POST /api/system/collect_agent_data       — incremental news / calendar / twitter data collection
GET  /api/agents/data-status              — MAX(date) per agent's SQLite archive
GET  /api/agents/twitter-auth-status      — check twitter_cookies.json session health
POST /api/agents/twitter-upload-cookies   — re-login without stopping API (requires TWITTER_UPLOAD_KEY)
```

Plus `GET /docs` (Swagger) and `GET /redoc`.

**Concurrency locks**: multiagent router uses `asyncio.Lock()` per resource — one `_prediction_lock` and one `_collection_locks[agent_name]` per agent — returning HTTP 409 if already running. Classic ML router uses `_train_lock` and `_dataset_refresh_lock` similarly.

**Predictor caching**: `classic_ml_predictions.py` caches `Predictor` instances in `_predictor_cache` (one per `model_name`). All Predictors share the single `SharedBaseDataCache` instance created in `api/main.py` at startup.

## Model Artifacts

Each model saves two files in `Classic_ml_model_solutions/Created_models_to_use/{config_name}/`:
- `model_{type}_{name}.joblib` — sklearn Pipeline (SimpleImputer → StandardScaler → LogisticRegression)
- `metrics_{type}_{name}.json` — features, quality metrics, threshold, cv averages

Metrics JSON fields:
- Config: `config_name`, `model_path`, `target`, `features`, `n_features`, `thr`, `best_metric`, `best_fold_idx`
- Best fold OOS: `auc`, `acc`, `precision`, `recall`, `f1`, `n_oos_samples`
- CV averages: `cv_avg_auc`, `cv_avg_acc`, `cv_avg_precision`, `cv_avg_recall`, `cv_avg_f1`

**Feature source of truth for prediction**: `metrics_*.json["features"]` (saved at training time). Fallbacks: `model.feature_names_in_` → `config.json["base_feats"]`.

## Data Sources (28 datasets in Classic_ml_model_solutions/Dataset_pipeline/Dataset_builder_pipeline.py)

| Category | Count | Source |
|----------|-------|--------|
| Futures OI (history, aggregated, stablecoin, coin-margin) | 4 | CoinGlass API (Bybit) |
| Futures Funding (history, OI-weight, vol-weight) | 3 | CoinGlass API |
| Futures Long/Short (global, top account, top position) | 3 | CoinGlass API (Bybit) |
| Futures Net Position v2 | 1 | CoinGlass API (Bybit) |
| Futures Liquidation (history, aggregated) | 2 | CoinGlass API (Bybit) |
| Futures Orderbook (ask/bids, aggregated) | 2 | CoinGlass API (Bybit) |
| Futures Taker Volume (v2, aggregated) | 2 | CoinGlass API (Bybit) |
| Exchange (Bitfinex margin, Coinbase premium, CGDI index) | 3 | CoinGlass API |
| On-chain (LTH supply, active addresses, STH supply, reserve risk) | 4 | CoinGlass API |
| Spot BTC OHLCV | 1 | CoinGlass API (Bybit) |
| S&P 500 OHLCV | 1 | yfinance (^GSPC) |
| Gold Futures OHLCV | 1 | yfinance (GC=F) |
| IGV Tech ETF OHLCV | 1 | yfinance (IGV) |

"Bybit" means the `exchange` / `exchange_list` query param is pinned to Bybit in both `Classic_ml_model_solutions/Dataset_pipeline/FeaturesGetterModule/features_endpoints.json` defaults and the explicit kwargs in `Classic_ml_model_solutions/Dataset_pipeline/Dataset_builder_pipeline.py` (`get_features`). To switch data source you must update BOTH files (there is no global `EXCHANGE` constant).

## Data Conventions

- All DataFrames have `date` column (datetime, YYYY-MM-DD)
- Feature columns use prefix pattern: `{source}__{metric}` (e.g., `futures_open_interest_history__close`)
- Derived features: `__diff1`, `__pct1` suffixes
- TA features: `{prefix}__ta_{indicator}` (e.g., `gold__ta_adx`, `sp500__ta_rsi`, `igv__ta_rsi`, `spot_price_history__ta_bbw`)
- 8 TA indicators per asset (4 assets: gold, sp500, igv, spot_price_history): `ta_adx`, `ta_cci`, `ta_rsi`, `ta_roc`, `ta_atr`, `ta_bbw`, `ta_obv`, `ta_mfi`
- Price MA features: `{col}__sma{7,14,21,50}`, `{col}__sma{N}_rel` (ratio), `{col}__zscore{7,14,50}`
- Lag features: `{col}__lag{1,3,5,7,15}` — applied to every non-diff/non-pct column in step 11 of the shared pipeline
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

## Config Structure (configs/config.json)

Array of 8 classic-ML experiment objects. Each has:
- `config_name`: e.g. `"base_model_1d"`, `"range_model_3d"`
- `N_DAYS`: prediction horizon (1, 3, 5, 7)
- `base_feats`: list of feature column names for this model
- `threshold`: probability threshold for binary classification
- `ma_window`: (range models only) SMA window for baseline, typically 7 or 14

`configs/ml_config.json` is an alternate/experimental feature-set file — not wired into the default training pipeline.

## Multiagent System

Located in `MultiagentSystem/`. Built on **LangGraph** — a DAG of LLM agents that each produce an `AgentSignal` (`prediction: bool`, `confidence: "high"|"medium"|"low"`, reasoning, risks), validated by a checker and merged by a reports analyser into a single LONG/SHORT verdict with a confidence score.

### Graph (`multiagent_system_main.py`)

```
START → supervisor → [agent_for_analysing_tech_indicators, agent_for_twitter_analysis]
                   → validator (fan-in)
                   → _should_retry?
                       ├─ retry → supervisor (if any agent has requirements and budget left)
                       └─ done  → agent_reports_analyser → END
```

Currently enabled nodes: `agent_for_analysing_tech_indicators`, `agent_for_twitter_analysis`, `validator`, `agent_reports_analyser`. News / on-chain / economic calendar agents exist as code but are commented out in the graph builder. `MAX_RETRIES = 2` per non-news agent.

### Key files

| File | Purpose |
|---|---|
| `multiagent_system_main.py` | LangGraph builder + `__main__` runner; exports compiled `app` |
| `multiagent_predictions_module.py` | `make_one_prediction`, `make_prediction_for_last_N_days`, `add_y_true`, `build_confusion_matrix` |
| `multiagent_types.py` | `AgentState` TypedDict, `AgentSignal`, `AgentRetry`, reducers (`merge_dicts`, `merge_retry_agents`) |
| `multiagent_config.json` | `forecast_start_date`, `horizon`, `agent_envolved_in_prediction`, per-agent settings (`window_to_analysis`, `base_feats`, Twitter authors/decay, etc.) |
| `hyperparameters_tuner.py` | Grid search over ranges for multiagent hyperparams, logs top-N to `tuning_top.json` |
| `predictions_results.csv` | Last batch-prediction output (standalone runner) |
| `confusion_matrix.png` | Last confusion matrix (standalone runner) |
| `agents/twitter_analyser/twitter_archive.db` | SQLite tweet archive |
| `agents/news_analyser/news_archive.json` | News archive |
| `agents/economic_calendar_analyser/` | Calendar archive + collector |

### Agents

- `agents/tech_indicators/agent_for_analysing_tech_indicators.py` — LLM reads windowed TA/OHLCV slice from the cached base df; system prompt at `agents/tech_indicators/system_prompt_general.md`.
- `agents/twitter_analyser/` — tweet collector (`twitter_scrapper/`), news classifier (`twitter_news_classifier/classifier.py`), and agent (`agent_for_twitter_analysis.py`) that applies time-decayed weights per author (`decay_rate`, `decay_start_day`, `initial_weight`). Authors come from `multiagent_config.json["agent_settings"]["agent_for_twitter_analysis"]["authors"]`.
- `agents/news_analyser/`, `agents/onchain_indicators/`, `agents/economic_calendar_analyser/` — exist but currently not wired into the graph.
- `agents/verdicts_validator/agent_for_verdicts_validation.py` — quality check on agent outputs; can request `recompose_report` which triggers a retry loop.
- `agents/reports_analyser/` — aggregates validated signals into the final LONG/SHORT + confidence.

### Twitter scraper auth flow

Chrome profile at `agents/twitter_analyser/twitter_scrapper/chrome_profile/`. Cookies persist across restarts in `twitter_cookies.json`. To re-login when the API is running: POST uploaded cookies to `/api/agents/twitter-upload-cookies` with `TWITTER_UPLOAD_KEY` env var. Manual alternative:
```bash
python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login
```
