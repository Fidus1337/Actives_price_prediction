# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin price direction prediction system with two independent prediction pipelines served over one FastAPI service:

1. **Classic ML** — Logistic Regression on CoinGlass + yfinance features. Horizons 1d/3d/5d/7d, two model types: **base** (price direction) and **range** (volatility above MA baseline).
2. **Multiagent system** — LangGraph DAG of LLM-powered agents (Twitter analyser, Tech indicators analyser, Verdicts validator, Reports analyser) that vote on LONG/SHORT direction for a requested `forecast_start_date`.

Market data comes from **CoinGlass (Bybit)** for futures/on-chain and yfinance for S&P 500 / Gold / IGV.

## Working Rules (Karpathy Principles)

These four rules are **mandatory**, override default behaviour, and apply to every task in this repo. They exist to counter the most common LLM failure modes: silent assumptions, over-engineering, drive-by edits, and unverified output.

### 1. Think Before Coding — never guess

If a request is ambiguous, under-specified, or missing context, **STOP**. Do NOT pick the most likely interpretation and start coding. Instead:

1. List the possible interpretations of the task in plain language.
2. State the concrete information you are missing (file path? expected behaviour? edge case? data shape?).
3. Ask the user one focused clarifying question and wait.

Examples that REQUIRE a question, not a guess:
- "fix the bug" without saying which one or where
- "add a new agent" without saying what it consumes or outputs
- "update the model" with multiple candidate files (`base_model_*`, `range_model_*`, `vol_scaled_*`, `ret_threshold_*`)
- a feature whose behaviour at a boundary (empty input, missing column, retry exhausted, NaN) is unspecified

Silently picking an interpretation and building 200 lines on top of it is the **worst** failure mode in this repo and is explicitly forbidden.

### 2. Simplicity First — minimum viable code only

Write the smallest amount of code that makes the current task work. Concretely:

- No abstractions "for future flexibility" — no base classes, no plugin systems, no config flags, no dependency injection unless the task requires it.
- No options/parameters the user didn't ask for. If a function needs one path today, give it one path. Don't add `mode="..."` "just in case".
- No premature generalisation. Three similar lines beat one clever helper. Wait until there are **three real** call sites before extracting.
- No defensive wrappers around code you control (try/except that re-raises, validations on internal-only inputs, `if x is None` for values that cannot be `None` per the type contract). Validate only at system boundaries (user input, external APIs, file/network I/O).
- No backwards-compatibility shims, no feature flags, no dead-code "commented-out for later". If the change supersedes old behaviour, delete the old behaviour.

When in doubt: 50 lines that solve today's problem beat 200 lines that solve a hypothetical future one.

### 3. Surgical Changes — touch only what the task demands

When editing existing files:

- Modify only the lines required by the task. Do not reformat surrounding code, rename unrelated variables, fix unrelated lint warnings, reorder imports beyond what your edit makes necessary, or "clean up" code you did not need to touch.
- Match the existing style of the file even if you disagree with it (indentation width, quote style, naming conventions, log-tag prefix format, comment language — Russian comments stay Russian, English stays English).
- The only "drive-by cleanup" allowed is removing **your own** newly-orphaned artifacts: imports made unused by your edit, variables your edit dropped, dead branches your edit made unreachable. Pre-existing unused imports / dead code are not yours to fix in this commit.
- Do not delete pre-existing commented-out code blocks. The author left them for a reason; if you think they should go, raise it as a question, do not delete unilaterally.
- When refactoring is genuinely needed, propose it first and wait for approval before mixing it into a feature/fix.

### 4. Verifiable Success Criteria — close the feedback loop

Every change must be **verifiable in a binary pass/fail way before you report it as done**. Concretely:

- For pure functions and data transformations: write a small script or pytest case that executes the function on representative input and asserts the expected output. If the assertion fails, the change is not done.
- For pipeline / agent / API changes: run the actual entry point end-to-end (`python -m ...`, `uvicorn ...`, the relevant `make_one_prediction(...)` call) and check the observable output against an expected shape or value. Type checks and dry-runs are NOT verification — they prove the code parses, not that it works.
- For UI changes: open the page in a browser and exercise the changed flow.
- If verification is impossible in the current sandbox (missing API keys, headed browser required, machine-specific data), say so **explicitly** instead of claiming success. The honest message is "implemented but not verified because X" — never report a task as complete based on "it should work" or "the diff looks right".
- Prefer fast, deterministic checks. A failing assertion that points at the broken line is worth more than a passing-by-luck integration run.

Order of operations: write the verification first (or at least sketch its expected pass condition), implement until it passes, then report.

---

These four rules supersede any general LLM defaults. If a section later in this file appears to conflict with them, these rules win.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train all classic ML models (reads configs/ml_config.json → saves to Classic_ml_model_solutions/Created_models_to_use/)
python -m Classic_ml_model_solutions.Models_builder_pipeline.Models_builder_pipeline

# Start API (port 8000) — serves both classic ML and multiagent endpoints
uvicorn api.main:app --reload

# Run multiagent system standalone (reads configs/multiagent_config.json)
python -m MultiagentSystem.multiagent_system_main

# Tune Twitter agent hyperparameters via Optuna (writes top-N runs to MultiagentSystem/agents_tuners/twitter_tuner/tuning_top.json)
python -m MultiagentSystem.agents_tuners.twitter_tuner.tuner_main

# Required env in dev.env:
# COINGLASS_API_KEY=...
# OPENAI_API_KEY=... (for multiagent LLM calls routed via MultiagentSystem/llm_factory.py)
# CLAUDE_KEY=... (optional, used when an agent's model id starts with "claude-")
# TWITTER_UPLOAD_KEY=... (optional, for /api/agents/twitter-upload-cookies)
# TWITTER_EMAIL=..., TWITTER_PASSWORD=... (used by chrome_login_before_scrapping.py for the headed re-login flow)
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
│   ├── multiagent_graph.py                 # build_multiagent_graph() — wires all agent nodes; exports compiled `app`
│   ├── multiagent_system_main.py           # __main__ runner; re-exports `app` from multiagent_graph
│   ├── multiagent_predictions_module.py    # make_one_prediction, make_prediction_for_last_N_days, add_y_true, build_confusion_matrix
│   ├── multiagent_types.py                 # AgentState TypedDict, AgentSignal, AgentRetry, NON_VALIDATED_AGENTS, reducers
│   ├── llm_factory.py                      # make_chat_llm() — routes claude-* to ChatAnthropic, else ChatOpenAI
│   ├── agents/                             # tech_indicators / twitter_analyser / news_analyser / onchain_indicators / economic_calendar_analyser / verdicts_validator / reports_analyser / unbias_agent
│   ├── agents_tuners/twitter_tuner/        # Optuna-based hyperparameter search for the twitter agent (tuner_main.py, twitter_optuna_tuner.py, twitter_tuner.py grid fallback, tuning_top.json output)
│   └── predictions_results*.csv            # Per-agent-combo snapshots from standalone runs (tech_twitter_v2, econ_twitter, twitter_onhain_tech, eco_twitter_tech, …)
├── Classic_ml_model_solutions/Dataset_pipeline/Dataset_builder_pipeline.py  # get_features() — fetches 28 datasets in parallel (ThreadPoolExecutor)
├── Classic_ml_model_solutions/Models_builder_pipeline/Models_builder_pipeline.py  # Training orchestrator: main_pipeline() per config in configs/ml_config.json
├── Classic_ml_model_solutions/PlotsBuilder/Plots_Builder.py  # ROC, metrics-vs-threshold, confusion matrix plots
├── new_targets.py                          # Experimental targets (triple barrier, vol-scaled, return-threshold)
├── configs/
│   ├── ml_config.json                      # Classic-ML training config — top-level {"runs": [...]} with 8 experiment objects
│   └── multiagent_config.json              # Multiagent runtime config (forecast_start_date, agents, agent_settings, neutral_threshold)
├── notebooks/                              # Jupyter experiments
├── Logs/available_features.json            # Auto-generated on every data fetch — ground truth for feature names
├── dev.env                                 # COINGLASS_API_KEY, OPENAI_API_KEY, CLAUDE_KEY, TWITTER_UPLOAD_KEY, TWITTER_EMAIL, TWITTER_PASSWORD
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
GET  /api/health                          — {"status": "healthy", "models_loaded": <bool dict per model>}
GET  /api/dataset-status                  — dataset load status, last_refreshed_at, shape
POST /api/system/train_classic_ml_models  — retrain classic ML models from configs/ml_config.json (or a custom payload)
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

**Feature source of truth for prediction**: `metrics_*.json["features"]` (saved at training time). Fallbacks: `model.feature_names_in_` → `ml_config.json["runs"][i]["base_feats"]`.

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
- TA features: 32 total (8 indicators × 4 assets) replace the old lag-based feature engineering
- Graphics saved to `graphics/{config_name}/`: ROC, metrics-vs-threshold, confusion matrix
- Logging: `LoggingSystem` redirects stdout to `logs.log` during training
- `available_features.json` auto-generated on each data fetch — ground truth for available features

## Config Structure (configs/ml_config.json)

Top-level wrapper: `{"runs": [...]}` — 8 classic-ML experiment objects loaded by `Models_builder_pipeline.load_config()` via `.get("runs", [])`. Each entry has:
- `name`: e.g. `"base_model_1d"`, `"range_model_3d"` (used as folder name and `CONFIG_NAME` in metrics JSON)
- `N_DAYS`: prediction horizon (1, 3, 5, 7)
- `base_feats`: list of feature column names for this model
- `threshold`: probability threshold for binary classification
- `ma_window`: (range models only) SMA window for baseline, typically 7 or 14
- `range_feats`: (range models only) extra range-target features added on top of `base_feats`

Both the CLI training entry point (`Models_builder_pipeline.py:__main__`) and the API router (`api/routers/classic_ml_predictions.py:CONFIG_PATH`) read this same file. The API also accepts a custom `runs` payload via `POST /api/system/train_classic_ml_models` to override the file.

## Multiagent System

Located in `MultiagentSystem/`. Built on **LangGraph** — a DAG of LLM agents that each produce an `AgentSignal` (`prediction: bool`, `confidence: "high"|"medium"|"low"`, reasoning, risks), validated by a checker and merged by a reports analyser into a single LONG/SHORT verdict with a confidence score.

### Graph (`MultiagentSystem/multiagent_graph.py`)

`build_multiagent_graph()` wires all five agent nodes — `agent_for_analysing_tech_indicators`, `agent_for_analysing_onchain_indicators`, `agent_for_news_analysis`, `agent_for_twitter_analysis`, `agent_for_economic_calendar_analysis` — in parallel from `supervisor`, fans them in to `validator`, then routes via `_should_retry` to either re-run flagged agents or hand off to `agent_reports_analyser`. `multiagent_system_main.py` only re-exports the compiled `app` and provides the `__main__` runner.

```
START → supervisor → [tech, onchain, news, twitter, economic_calendar]   (fan-out, parallel)
                   → validator                                           (fan-in)
                   → _should_retry?
                       ├─ retry → supervisor (any agent has requirements & retry budget left)
                       └─ done  → agent_reports_analyser → END
```

**Which agents actually contribute votes is decided at runtime**, not by the graph: every agent function checks `state["agent_envolved_in_prediction"]` (sourced from `multiagent_config.json`) and short-circuits with `return {}` if its name is absent. Inactive nodes still execute but produce no signal.

`MAX_RETRIES = 2` per retry-eligible agent (everything except `multiagent_types.NON_VALIDATED_AGENTS`).

### Key files

| File | Purpose |
|---|---|
| `multiagent_graph.py` | `build_multiagent_graph()` — node registration, fan-out/fan-in edges, `_should_retry` router; exports compiled `app` |
| `multiagent_system_main.py` | `__main__` runner that re-exports `app` from `multiagent_graph` |
| `multiagent_predictions_module.py` | `make_one_prediction`, `make_prediction_for_last_N_days`, `add_y_true`, `build_confusion_matrix` |
| `multiagent_types.py` | `AgentState` TypedDict, `AgentSignal`, `AgentRetry`, `NON_VALIDATED_AGENTS`, reducers (`merge_dicts`, `merge_retry_agents`) |
| `llm_factory.py` | `make_chat_llm(model, temperature, **kwargs)` — routes `claude-*` ids to `ChatAnthropic` (uses `CLAUDE_KEY`), everything else to `ChatOpenAI` (uses `OPENAI_API_KEY`) |
| `multiagent_config.json` | `forecast_start_date`, `horizon`, `agent_envolved_in_prediction`, `neutral_threshold`, per-agent `agent_settings` (`window_to_analysis`, `base_feats`, Twitter authors/decay, etc.) |
| `agents_tuners/twitter_tuner/tuner_main.py` | Optuna entry point for Twitter agent hyperparam search; writes top-N to `agents_tuners/twitter_tuner/tuning_top.json`. `twitter_tuner.py` keeps an older grid-search fallback. |
| `predictions_results.csv` (and `predictions_results_*.csv`) | Per-agent-combo snapshots from standalone runs (one CSV per saved experiment) |
| `confusion_matrix.png` | Last confusion matrix (standalone runner) |
| `agents/twitter_analyser/twitter_archive.db` | SQLite tweet archive |
| `agents/news_analyser/news_archive.json` | News archive |
| `agents/economic_calendar_analyser/` | Calendar archive + collector |

### Agents

- `agents/tech_indicators/agent_for_analysing_tech_indicators.py` — LLM reads windowed TA/OHLCV slice from the cached base df; system prompt at `agents/tech_indicators/system_prompt_general.md`.
- `agents/twitter_analyser/` — tweet collector (`twitter_scrapper/`), tweet classifier (`twitter_news_classifier/classifier.py`, LLM-based, runs at collection time and emits `signal_type ∈ {BULL, BEAR, NO_CORRELATION_TO_BTC}` × `signal_confidence ∈ {LOW, MIDDLE, HIGH}`), and aggregation agent (`agent_for_twitter_analysis.py`) that applies a per-day exponential decay (see "Twitter aggregation pipeline" below). Authors come from `multiagent_config.json["agent_settings"]["agent_for_twitter_analysis"]["authors"]`.
- `agents/news_analyser/`, `agents/onchain_indicators/`, `agents/economic_calendar_analyser/` — wired into the graph but inactive unless listed in `agent_envolved_in_prediction`.
- `agents/verdicts_validator/agent_for_verdicts_validation.py` — quality check on agent outputs; can request `recompose_report` which triggers a retry loop.
- `agents/reports_analyser/` — aggregates validated signals into the final LONG/SHORT + confidence.

### Reports analyser — final verdict

`agents/reports_analyser/agent_for_reports_analysis.py:compute_confidence_score()` produces the final aggregate. Each voting agent contributes:

```
weight  = {"low": 1, "medium": 2, "high": 3}[signal["confidence"]]
sign    = +1 if signal["prediction"] is True (LONG/HIGHER) else -1
vote    = sign * weight                               # ∈ {-3, -2, -1, +1, +2, +3}
score   = arithmetic mean of votes over voting agents # ∈ [-3, +3]
```

Agents with `prediction is None` or `confidence is None` (stub agents, or formula-based agents that returned "no actionable signal") abstain and are excluded from the mean.

`direction` is then decided by `multiagent_config.json["neutral_threshold"]`:
- `score > neutral_threshold` → `LONG`
- `score < -neutral_threshold` → `SHORT`
- otherwise → `None` (neutral)

**Default `neutral_threshold` is `0.0`**, which means there is **no neutral band** — any non-zero score produces a LONG/SHORT verdict. Raise it (e.g. to `1.0`) to filter out low-conviction sells.

API serialization in `api/routers/multiagent_predictions.py`: `_DIRECTION_MAP = {"LONG": 1, "SHORT": 0}`; `None` becomes `null`. The float `score` is exposed as `confidence_score` in the response.

### Twitter aggregation pipeline (`agent_for_twitter_analysis.py`)

The agent does NOT call an LLM at prediction time — it reads pre-classified tweets from `twitter_archive.db` and aggregates in four steps for the window `[forecast_start_date - window_to_analysis + 1, forecast_start_date]`:

1. **Group by date** — tweets within the window, after filtering to configured `authors` and dropping `signal_type ∈ {"", "NO_CORRELATION_TO_BTC"}`.
2. **Per-author per-date averaging** — for each `(date, author)`, average per-tweet signed scores (BULL: +1/+2/+3, BEAR: −1/−2/−3 by `LOW/MIDDLE/HIGH`); the author's daily vote = `(round(abs(avg)), sign(avg))`. Authors averaging to 0 are dropped for that date.
3. **Authors → one signal per date** — average signed author votes within each date; ties round to 0 and drop the date.
4. **Date signals → final verdict with age decay**, relative to `forecast_start_date`:
   - `age < decay_start_day` → weight = `1.0` (fresh zone)
   - `age >= decay_start_day` → weight = `initial_weight * (1 - decay_rate) ** (age - decay_start_day)`

   Compute weighted average of signed daily scores; final `signal_confidence = round(abs(avg))`. If 0, agent abstains (returns `prediction=None`, `confidence=None`).

### `predictions_results.csv` schema

Written by `make_one_prediction` / `make_prediction_for_last_N_days` in `multiagent_predictions_module.py`. Columns:

- `forecast_start_date`, `y_predict` (`"LONG"` / `"SHORT"` / `None`), `y_predict_confidence` (= `confidence_score`, float in `[-3, +3]`), `summary`, `reasoning`, `risks`
- Per-agent flatten: `{agent_short}__prediction`, `{agent_short}__confidence` — where `agent_short` = `agent_name` with `agent_for_` and `agent_for_analysing_` prefixes stripped (e.g. `tech_indicators__prediction`, `twitter_analysis__confidence`)
- After `add_y_true()`: `start_date_price`, `btc_bybit_close_price`, `btc_bybit_high_price`, `btc_bybit_low_price`, `y_true`

### Twitter scraper auth flow

Chrome profile at `agents/twitter_analyser/twitter_scrapper/chrome_profile/`. Cookies persist across restarts in `twitter_cookies.json`. To re-login when the API is running: POST uploaded cookies to `/api/agents/twitter-upload-cookies` with `TWITTER_UPLOAD_KEY` env var. Manual alternative:
```bash
python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login
```
