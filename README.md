# BTC Price Direction Prediction

Bitcoin price direction prediction system with **two parallel pipelines** served by one FastAPI app:

1. **Classic ML** — sklearn LogisticRegression on CoinGlass + yfinance features. Four model families × four horizons (1d/3d/5d/7d) = 16 trained models.
2. **Multiagent system** — LangGraph DAG of LLM-powered agents (Twitter, tech indicators, news, on-chain, economic calendar) that vote on LONG/SHORT for a requested date.

Both pipelines share a single in-memory feature cache (`SharedBaseDataCache`) so they always see the same input data.

> **Port: `8080`** everywhere — `Dockerfile`, `docker-compose.yml`, local dev. All examples and curl snippets in this README hit `http://localhost:8080`.

---

## Table of Contents

1. [Quickstart (Docker)](#1-quickstart-docker)
2. [Local dev install](#2-local-dev-install)
3. [Environment variables](#3-environment-variables)
4. [API reference](#4-api-reference)
5. [Module CLI cheatsheet](#5-module-cli-cheatsheet)
6. [Configuration files](#6-configuration-files)
7. [Classic ML pipeline](#7-classic-ml-pipeline)
8. [Multiagent system](#8-multiagent-system)
9. [Project structure](#9-project-structure)
10. [Troubleshooting](#10-troubleshooting)
11. [Multiagent tests](#11-multiagent-tests)

---

## 1. Quickstart (Docker)

The fastest path to a running API. From a folder containing `dev.env` and the four mounted SQLite/profile files (see `docker-compose.yml`):

```bash
docker compose up -d
```

The container exposes the API on **`http://localhost:8080`**. Verify:

```bash
# Health
curl http://localhost:8080/api/health

# Swagger UI
open http://localhost:8080/docs
```

Required side files (mounted into the container by `docker-compose.yml`):

| Path on host | Mounted to | Used by |
|---|---|---|
| `dev.env` | `/app/dev.env` | API key + secrets |
| `news_archive.db` | `…/news_analyser/news_archive.db` | News agent |
| `calendar_archive.db` | `…/economic_calendar_analyser/calendar_archive.db` | Calendar agent |
| `twitter_archive.db` | `…/twitter_scrapper/twitter_archive.db` | Twitter agent |
| `chrome_profile/` | `…/twitter_scrapper/chrome_profile` | Twitter login session |

---

## 2. Local dev install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

Run the API on port **8080**:

```bash
# Production-shaped (matches Dockerfile)
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Dev (auto-reload on code change)
uvicorn api.main:app --reload --port 8080
```

> Always pass `--port 8080` explicitly — uvicorn's bare default (8000) is **not** what this project uses.

After launch:

- Swagger UI: <http://localhost:8080/docs>
- ReDoc: <http://localhost:8080/redoc>

---

## 3. Environment variables

Loaded from `dev.env` at the project root via `python-dotenv` ([api/main.py:33](api/main.py#L33)).

| Variable | Required | Where used | What breaks if missing |
|---|:---:|---|---|
| `COINGLASS_API_KEY` | yes | All CoinGlass data fetches | API starts but Classic ML and tech/onchain agents abstain (warning at startup) |
| `OPENAI_API_KEY` | yes (if any OpenAI agent active) | `MultiagentSystem/llm_factory.py` for `gpt-*` model ids | LLM agents return abstain stub (`429 insufficient_quota` or `RuntimeError`) |
| `CLAUDE_KEY` | optional | Same factory, for `claude-*` model ids | RuntimeError only when an agent is configured with a `claude-*` `llm_model` |
| `TWITTER_EMAIL`, `TWITTER_PASSWORD` | required for headed re-login | `chrome_login_before_scrapping.py` | Re-login flow fails (existing cookies still work) |
| `TWITTER_USERNAME` | optional | Same — fallback for "unusual activity" challenge | Verification challenge cannot be solved |
| `TWITTER_UPLOAD_KEY` | required for `/api/agents/twitter-upload-cookies` | [`api/routers/multiagent_predictions.py:249`](api/routers/multiagent_predictions.py#L249) | Endpoint always returns 401 |

Example `dev.env`:

```env
COINGLASS_API_KEY=...
OPENAI_API_KEY=sk-...
CLAUDE_KEY=sk-ant-...
TWITTER_EMAIL=you@example.com
TWITTER_PASSWORD=...
TWITTER_USERNAME=your_handle
TWITTER_UPLOAD_KEY=<exactly 100 random chars>
```

---

## 4. API reference

All endpoints are mounted under the prefix `/api` in [`api/main.py`](api/main.py). Full request/response schemas live in [`api/schemas.py`](api/schemas.py); Swagger at `/docs` is the live source of truth.

### 4.1 Endpoint table

| Method | Path | Purpose | Concurrency |
|---|---|---|---|
| GET | `/api/health` | Server status + which models are cached | — |
| GET | `/api/dataset-status` | Whether `SharedBaseDataCache` is loaded; shape and last refresh | — |
| **Classic ML** | | | |
| POST | `/api/predictions` | Batch predict by `(models[], dates[])`; optional `refresh_dataset` | shares `_dataset_refresh_lock` (409 if a refresh is running) |
| GET | `/api/models` | List trained models with `cv_avg_*` quality metrics | — |
| POST | `/api/system/train_classic_ml_models` | Retrain from `configs/ml_config.json` or a custom `runs[]` body | `_train_lock` → 409 if already running |
| **Multiagent** | | | |
| POST | `/api/multiagent_predictions` | Run LangGraph DAG for last N eligible dates | `_prediction_lock` → 409 if already running |
| POST | `/api/system/collect_agent_data` | Incremental fetch into news / calendar / twitter SQLite archives | per-agent `_collection_locks` → 409 per agent |
| GET | `/api/agents/data-status` | MAX(date) per agent's SQLite archive | — |
| GET | `/api/agents/twitter-auth-status` | Twitter session + cookie health check | — |
| POST | `/api/agents/twitter-upload-cookies` | Replace `twitter_cookies.json` (re-login without restart) | requires `TWITTER_UPLOAD_KEY` (401 otherwise) |

### 4.2 Concurrency model

Long-running endpoints are guarded by `asyncio.Lock` instances so that **only one** instance of each operation can run at a time. The currently-running request is **never interrupted** — it runs to completion. Any **new** request that arrives while the lock is held is **rejected immediately with HTTP 409 Conflict** (no queuing, no waiting). The client decides whether to retry later.

Example: a multiagent run for 100 dates may take ~1 hour. While it runs, a second `POST /api/multiagent_predictions` returns 409 instantly. The first run keeps going untouched. As soon as it finishes, the next call is accepted.

Locks:

- `_dataset_refresh_lock` — held during `SharedBaseDataCache.refresh()` ([classic_ml_predictions.py](api/routers/classic_ml_predictions.py))
- `_train_lock` — held during classic-ML training
- `_prediction_lock` — held for the duration of a multiagent run
- `_collection_locks[agent]` — one per agent (`news_analyser`, `economic_calendar_analyser`, `twitter_analyser`)

### 4.3 Examples

**Classic ML — predict for two models on two dates:**

```bash
curl -X POST "http://localhost:8080/api/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["base_model_1d", "range_model_3d"],
    "dates":  ["2026-04-20", "2026-04-21"],
    "refresh_dataset": false
  }'
```

Response shape (abridged):

```json
{
  "requested_models": ["base_model_1d", "range_model_3d"],
  "requested_dates":  ["2026-04-20", "2026-04-21"],
  "results": [
    {
      "model_name": "base_model_1d",
      "model_type": "base",
      "horizon_days": 1,
      "found_dates": ["2026-04-20"],
      "missing_dates": ["2026-04-21"],
      "predictions": [
        {"date": "2026-04-20", "prediction": 1, "probability": 0.654, "spot_price_close": 70509.9}
      ],
      "error": null
    }
  ]
}
```

**Classic ML — list models with metrics:**

```bash
curl http://localhost:8080/api/models
```

**Classic ML — retrain from `configs/ml_config.json`:**

```bash
curl -X POST "http://localhost:8080/api/system/train_classic_ml_models"
```

**Classic ML — retrain from a custom payload:**

```bash
curl -X POST "http://localhost:8080/api/system/train_classic_ml_models" \
  -H "Content-Type: application/json" \
  -d '{"runs": [{"name": "base_model_1d", "N_DAYS": 1, "threshold": 0.5,
                  "base_feats": ["spot_price_history__close__pct1"]}]}'
```

**Multiagent — run for last 10 eligible dates:**

```bash
curl -X POST "http://localhost:8080/api/multiagent_predictions" \
  -H "Content-Type: application/json" \
  -d @configs/multiagent_config.json
# or send a body with the same shape plus n_last_dates
```

The body schema mirrors `configs/multiagent_config.json` plus `n_last_dates: int` (1–365). See [`api/schemas.py:MultiagentPredictionsRequest`](api/schemas.py) for the full Pydantic example.

**Multiagent — collect news + calendar + twitter:**

```bash
curl -X POST "http://localhost:8080/api/system/collect_agent_data" \
  -H "Content-Type: application/json" \
  -d '{"agents": ["news_analyser", "economic_calendar_analyser"]}'
```

**Multiagent — Twitter session health and re-login:**

```bash
# Check if cookies are still good
curl http://localhost:8080/api/agents/twitter-auth-status

# Upload fresh cookies (when relogin_required=true and you can't open a GUI on the host)
curl -X POST "http://localhost:8080/api/agents/twitter-upload-cookies" \
  -H "Content-Type: application/json" \
  -d '{"upload_key": "<TWITTER_UPLOAD_KEY value>",
       "cookies":    [{"name": "auth_token", "value": "...", "domain": ".x.com", "path": "/"},
                      {"name": "ct0",        "value": "...", "domain": ".x.com", "path": "/"}]}'
```

### 4.4 Python client snippet

```python
import requests

BASE = "http://localhost:8080"

resp = requests.post(f"{BASE}/api/predictions", json={
    "models": ["base_model_1d", "range_model_3d"],
    "dates":  ["2026-04-20"],
})
for result in resp.json()["results"]:
    if result["error"]:
        print(f"{result['model_name']}: ERROR — {result['error']}")
        continue
    for pred in result["predictions"]:
        direction = "UP" if pred["prediction"] == 1 else "DOWN"
        print(f"{result['model_name']} {pred['date']}: {direction} (p={pred['probability']:.4f})")
```

---

## 5. Module CLI cheatsheet

All commands are run from the project root with the venv activated.

| Task | Command |
|---|---|
| Train all classic-ML models from `configs/ml_config.json` | `python -m Classic_ml_model_solutions.Models_builder_pipeline.Models_builder_pipeline` |
| Run multiagent predictions for last N days (config-driven) | `python -m MultiagentSystem.multiagent_system_main` |
| Tune Twitter agent hyperparameters via Optuna | `python -m MultiagentSystem.agents_tuners.twitter_tuner.tuner_main` |
| Re-login to Twitter (headed Chrome, writes `twitter_cookies.json`) | `python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login` |
| Start the API locally (canonical port) | `uvicorn api.main:app --host 0.0.0.0 --port 8080` |
| Start the API for development | `uvicorn api.main:app --reload --port 8080` |

---

## 6. Configuration files

### 6.1 `configs/ml_config.json` — Classic ML

Top-level wrapper: `{"runs": [...]}` — one entry per trained model.

| Field | Type | Required | Description |
|---|---|:---:|---|
| `name` | str | yes | Determines model family by **substring** (`base_model_*`, `range_model_*`, `ret_threshold_model_*`, `vol_scaled_model_*`). Also used as the output folder name. |
| `N_DAYS` | int | yes | Forecast horizon in days (1, 3, 5, 7). |
| `threshold` | float | no | Probability threshold for binary class (default `0.5`). |
| `base_feats` | list[str] | yes | Feature columns to train on. |
| `ma_window` | int | range models only | SMA window for the volatility baseline. |
| `range_feats` | list[str] | range models only | Extra range-target features (`range_pct`, `range_pct_ma{W}`). |

Feature naming: `{source}__{metric}__{suffix}` — for example `futures_open_interest_aggregated_history__close__pct1`. Suffixes:

- `__diff1` — first difference (today − yesterday)
- `__pct1` — daily percent change
- `__lag{N}` — value shifted N days back (`N ∈ {1, 3, 5, 7, 15}`)
- `__sma{N}_rel`, `__zscore{N}` — added by the price-MA step

### 6.2 `configs/multiagent_config.json` — Multiagent

```json
{
  "forecast_start_date": "2026-04-20",
  "horizon": 1,
  "agent_envolved_in_prediction": [
    "agent_for_twitter_analysis",
    "agent_for_analysing_tech_indicators"
  ],
  "neutral_threshold": 0.0,
  "agent_settings": {
    "agent_for_analysing_tech_indicators": {
      "system_prompt_file": "agents/tech_indicators/system_prompt_general.md",
      "llm_model": "gpt-4.1",
      "window_to_analysis": 21,
      "base_feats": ["spot_price_history__close", "..."]
    },
    "agent_for_twitter_analysis": {
      "authors": ["CarpeNoctom", "rektcapital", "..."],
      "window_to_analysis": 14,
      "decay_rate": 0.05,
      "decay_start_day": 1,
      "initial_weight": 1.0
    },
    "verdicts_validator": { "llm_model": "gpt-4.1" }
  }
}
```

- `agent_envolved_in_prediction` — only agents listed here vote. Others run the node but return `{}`.
- `neutral_threshold` — score in `[-3, +3]`; `|score| ≤ neutral_threshold` → verdict `None`. Default `0.0` means no neutral band.
- `agent_settings` — per-agent block, schema-free. Common keys: `llm_model`, `system_prompt_file`, `window_to_analysis`, `base_feats`, `decay_rate`, `decay_start_day`, `initial_weight`, `authors`.

---

## 7. Classic ML pipeline

### 7.1 Shared base data pipeline (`SharedBaseDataCache`)

Runs once, cached in memory, thread-safe with `threading.Lock` and a TTL of 3600s. Used by classic-ML training, classic-ML inference, and the multiagent system — single source of truth for features.

```
1.  get_features() → 28 DataFrames           (ThreadPoolExecutor, max_workers=8)
2.  outer-merge by date, dedupe="last"
3.  ensure_spot_prefix()
4.  ffill all feature columns
5.  Keep last _DATE_WINDOW_DAYS=1000 days
6.  Drop 12 hardcoded _SPARSE_COLUMNS
7.  Re-ffill + dropna
8.  add_engineered_features()                ← __diff1, __pct1, imbalance feats
9.  add_price_ma_features()                  ← SMA 7/14/21/50, __smaN_rel, __zscoreN
10. add_ta_features_selected(...)            ← 8 TA indicators × 4 assets (gold, sp500, igv, spot)
11. Lag every non-diff/non-pct column by    _LAG_PERIODS=[1, 3, 5, 7, 15]
12. dropna() → trim to longest continuous segment
13. Write Logs/available_features.json
```

> No "drop columns with >30% NaN" rule — sparse columns are explicitly listed in `_SPARSE_COLUMNS` (12 columns: orderbook USD sides + CGDI index).

### 7.2 28 data sources

CoinGlass (Bybit-pinned in `features_endpoints.json` and `Dataset_builder_pipeline.get_features`) for futures, on-chain and exchange feeds; yfinance for S&P 500 (`^GSPC`), Gold (`GC=F`) and the IGV tech ETF.

| Category | Count |
|---|:---:|
| Futures Open Interest (history, aggregated, stablecoin, coin-margin) | 4 |
| Futures Funding (history, OI-weighted, vol-weighted) | 3 |
| Futures Long/Short (global, top accounts, top positions) | 3 |
| Futures Net Position v2 | 1 |
| Futures Liquidation (history, aggregated) | 2 |
| Futures Orderbook (ask/bids, aggregated) | 2 |
| Futures Taker Volume (v2, aggregated) | 2 |
| Exchange (Bitfinex margin, Coinbase premium, CGDI index) | 3 |
| On-chain (LTH, STH, active addresses, reserve risk) | 4 |
| Spot BTC OHLCV | 1 |
| S&P 500, Gold, IGV (yfinance) | 3 |
| **Total** | **28** |

### 7.3 Walk-forward CV + sklearn pipeline

```
Fold 1:  [===TRAIN===] [=TEST=]
Fold 2:  [=====TRAIN=====] [=TEST=]
Fold 3:  [========TRAIN========] [=TEST=]
Fold 4:  [==========TRAIN==========] [=TEST=]
```

Each fold trains:

```
SimpleImputer(strategy="mean") → StandardScaler() → LogisticRegression(max_iter=3000, class_weight="balanced")
```

Best model per config is selected by `best_metric` (default **accuracy**) across folds; `cv_avg_*` are stored alongside best-fold metrics in `metrics_*.json`.

### 7.4 Model artifacts

Output folder: `Classic_ml_model_solutions/Created_models_to_use/{config_name}/`

```
{config_name}/
├── model_{type}_{name}.joblib       # sklearn Pipeline
└── metrics_{type}_{name}.json       # features + threshold + best-fold + cv_avg metrics
```

Currently on disk (16 trained models = 4 families × 4 horizons):

```
base_model_{1,3,5,7}d/             # price direction
range_model_{1,3,5,7}d/            # volatility above MA baseline
ret_threshold_model_{1,3,5,7}d/    # return-threshold target
vol_scaled_model_{1,3,5,7}d/       # vol-scaled return target
```

### 7.5 `metrics_*.json` shape

```json
{
  "config_name": "base_model_1d",
  "model_path": "Classic_ml_model_solutions/Created_models_to_use/base_model_1d/model_base_base_model_1d.joblib",
  "target": "y_up_1d",
  "features": ["..."],
  "n_features": 21,
  "thr": 0.5,
  "best_metric": "accuracy",
  "best_fold_idx": 2,
  "auc": 0.5714,    "acc": 0.5561,
  "precision": 0.6, "recall": 0.6286, "f1": 0.614,
  "n_oos_samples": 248,
  "cv_avg_auc": 0.560, "cv_avg_acc": 0.548,
  "cv_avg_precision": 0.582, "cv_avg_recall": 0.611, "cv_avg_f1": 0.596
}
```

---

## 8. Multiagent system

A LangGraph DAG of LLM agents that vote on LONG/SHORT for a given `forecast_start_date`. Source under `MultiagentSystem/`.

### 8.1 Graph

```
START → supervisor → [tech, onchain, news, twitter, economic_calendar]   (parallel)
                   → validator                                           (fan-in)
                   → _should_retry?
                       ├─ retry → supervisor   (any agent has requirements & retry budget left)
                       └─ done  → reports_analyser → END
```

Active voting agents are decided **at runtime** from `agent_envolved_in_prediction`. Agents not listed there still execute but return `{}`. `MAX_RETRIES = 2` per retry-eligible agent.

### 8.2 Agent inventory

| Agent | What it does |
|---|---|
| `agent_for_analysing_tech_indicators` | LLM reads windowed TA/OHLCV slice from the cached base df; returns LONG/SHORT + confidence |
| `agent_for_analysing_onchain_indicators` | LLM analyses LTH/STH supply, active addresses, reserve risk |
| `agent_for_news_analysis` | Classifies news from `news_archive.db` and decays by age |
| `agent_for_twitter_analysis` | Aggregates pre-classified tweets from `twitter_archive.db`; **no LLM at predict-time** — pure formula with exponential age decay |
| `agent_for_economic_calendar_analysis` | LLM analyses major + medium US calendar events |
| `agent_for_verdicts_validation` | Quality-checks tech + onchain outputs; can request a retry |
| `agent_reports_analyser` | Aggregates validated signals: `score = mean(sign × weight)`; verdict by `neutral_threshold` |

### 8.3 Final verdict math

```
weight = {"low": 1, "medium": 2, "high": 3}[confidence]
sign   = +1 if prediction is True (LONG/HIGHER) else -1
vote   = sign * weight                               # ∈ {-3..-1, +1..+3}
score  = arithmetic mean of votes over voting agents # ∈ [-3, +3]

direction = LONG  if score >  neutral_threshold
            SHORT if score < -neutral_threshold
            None  otherwise
```

Agents with `prediction is None` or `confidence is None` abstain and are excluded from the mean.

### 8.4 `predictions_results.csv` schema

Written by `make_one_prediction` / `make_prediction_for_last_N_days` ([`MultiagentSystem/multiagent_predictions_module.py`](MultiagentSystem/multiagent_predictions_module.py)).

| Column | Meaning |
|---|---|
| `forecast_start_date` | Anchor date (YYYY-MM-DD) |
| `y_predict` | `"LONG"` / `"SHORT"` / `None` |
| `y_predict_confidence` | Aggregate score, float in `[-3, +3]` |
| `summary`, `reasoning`, `risks` | Human-readable verdict text |
| `{agent_short}__prediction` | Per-agent True (LONG) / False (SHORT) / None |
| `{agent_short}__confidence` | Per-agent `"high"`/`"medium"`/`"low"` / None |
| `start_date_price`, `btc_bybit_close_price`, `btc_bybit_high_price`, `btc_bybit_low_price` | Filled by `add_y_true()` |
| `y_true` | `"LONG"` / `"SHORT"` / `None` (None for future dates) |

`agent_short` = agent name with `agent_for_` and `agent_for_analysing_` stripped (e.g. `tech_indicators__prediction`).

---

## 9. Project structure

```
.
├── api/                                 # FastAPI service (serves both pipelines)
│   ├── main.py                          # App entry, CORS, lifespan
│   ├── schemas.py                       # Pydantic request/response models
│   └── routers/
│       ├── classic_ml_predictions.py    # /api/predictions, /api/models, /api/dataset-status,
│       │                                # /api/system/train_classic_ml_models, /api/health
│       └── multiagent_predictions.py    # /api/multiagent_predictions, /api/system/collect_agent_data,
│                                        # /api/agents/{data-status,twitter-auth-status,twitter-upload-cookies}
│
├── Classic_ml_model_solutions/
│   ├── Dataset_pipeline/
│   │   ├── Dataset_builder_pipeline.py  # get_features() — fetches 28 datasets in parallel
│   │   ├── FeaturesGetterModule/        # CoinGlass + yfinance client
│   │   ├── FeaturesEngineer/            # diff1/pct1, MA, TA features
│   │   └── SharedDataCache/             # SharedBaseDataCache — single shared feature cache
│   ├── Filtering_features_pipeline/
│   │   └── CorrelationsAnalyzer/
│   ├── Models_builder_pipeline/
│   │   ├── Models_builder_pipeline.py   # Training orchestrator (CLI entry)
│   │   └── ModelsTrainer/               # base / range / ret_threshold / vol_scaled trainers
│   ├── PlotsBuilder/                    # ROC, metrics-vs-threshold, confusion matrix plots
│   ├── Predict_with_ml_model/
│   │   └── Predictor.py                 # Inference class (consumed by /api/predictions)
│   └── Created_models_to_use/           # Trained model artifacts (16 folders)
│
├── MultiagentSystem/
│   ├── multiagent_graph.py              # build_multiagent_graph() — DAG wiring
│   ├── multiagent_system_main.py        # __main__ runner; re-exports compiled `app`
│   ├── multiagent_predictions_module.py # make_one_prediction / make_prediction_for_last_N_days
│   ├── multiagent_types.py              # AgentState, AgentSignal, reducers
│   ├── llm_factory.py                   # gpt-* → ChatOpenAI, claude-* → ChatAnthropic
│   ├── agents/                          # tech_indicators, twitter_analyser, news_analyser,
│   │                                    # onchain_indicators, economic_calendar_analyser,
│   │                                    # verdicts_validator, reports_analyser, unbias_agent
│   └── agents_tuners/twitter_tuner/     # Optuna hyperparameter search
│
├── configs/
│   ├── ml_config.json                   # Classic ML training runs[]
│   └── multiagent_config.json           # Multiagent runtime config
│
├── Logs/
│   ├── LoggingSystem/                   # stdout → logs.log helper (training/tuning only)
│   └── available_features.json          # Auto-generated on every data fetch
│
├── notebooks/                           # Jupyter experiments
├── graphics/{config_name}/              # ROC / metrics / confusion matrix per trained model
├── Dockerfile                           # Builds the prod image (port 8080)
├── docker-compose.yml                   # One-shot local deploy
├── dev.env                              # API keys + secrets (NOT committed in production)
└── requirements.txt
```

---

## 10. Troubleshooting

**`409 Conflict` on training / prediction / collection.** Another request is already holding the lock for that operation. Wait for it to finish; the API does not queue.

**Startup warning: "Failed to fetch CoinGlass dataset".** `COINGLASS_API_KEY` is missing or the API is unreachable. The server still starts but `/api/predictions` will fail until the cache loads. Check `GET /api/dataset-status` to confirm.

**Tech-indicators agent abstains every day with `no vote (skipped)`.** The LLM call failed (commonly `429 insufficient_quota` for OpenAI). Either top up the OpenAI account or change `agent_settings.agent_for_analysing_tech_indicators.llm_model` to a `claude-*` id (requires `CLAUDE_KEY`).

**`relogin_required: true` from `/api/agents/twitter-auth-status`.** The Twitter session expired. Two options:
1. Run the headed login on a machine with a display:
   ```bash
   python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login
   ```
2. Or upload fresh cookies via `POST /api/agents/twitter-upload-cookies` (needs `TWITTER_UPLOAD_KEY`).

**Chrome volume mount issues in Docker.** The container needs `chrome_profile/` mounted at the path shown in `docker-compose.yml`. Without it, every Twitter scrape starts from a fresh profile and fails authentication.

**`/api/system/train_classic_ml_models` returns 500.** Check `logs.log` (stdout is teed there during training). Common causes: a feature in `base_feats` is missing from the dataset, or the SMA window in `range_feats` doesn't match `ma_window`.

---

## 11. Multiagent tests

Tests for the multiagent system live in [`MultiagentSystem/tests/`](MultiagentSystem/tests) and are written with the stdlib **`unittest`** framework — no extra dependency required, just the venv from §2.

| File | What it covers | Type |
|---|---|---|
| [`test_reports_analyser.py`](MultiagentSystem/tests/test_reports_analyser.py) | `compute_confidence_score`: weight table, abstain handling, neutral threshold, breakdown text | Pure unit |
| [`test_twitter_aggregation.py`](MultiagentSystem/tests/test_twitter_aggregation.py) | The four pure aggregation helpers in `agent_for_twitter_analysis` (window dates, group-by-date, per-author averaging, age-decay verdict) | Pure unit |
| [`test_make_one_prediction.py`](MultiagentSystem/tests/test_make_one_prediction.py) | End-to-end run of `make_one_prediction` through the full LangGraph DAG with only the Twitter agent enabled (no LLM call, no network) | Integration |

### Run all multiagent tests

From the project root, with the venv activated:

```bash
# Windows
.venv\Scripts\python.exe -m unittest discover -s MultiagentSystem/tests -v

# macOS / Linux
.venv/bin/python -m unittest discover -s MultiagentSystem/tests -v
```

### Run a single test file

```bash
python -m unittest MultiagentSystem.tests.test_reports_analyser -v
python -m unittest MultiagentSystem.tests.test_twitter_aggregation -v
python -m unittest MultiagentSystem.tests.test_make_one_prediction -v
```

### Run a single test class or method

```bash
# Class
python -m unittest MultiagentSystem.tests.test_reports_analyser.TestComputeConfidenceScore -v

# Method
python -m unittest MultiagentSystem.tests.test_reports_analyser.TestComputeConfidenceScore.test_no_signals_returns_zero_neutral -v
```

### Notes

- The tests make no network calls and never invoke an LLM. `test_make_one_prediction.py` patches the SQLite tweet reader via `unittest.mock`, and the validator short-circuits on `claude-*` models when `CLAUDE_KEY` is unset. They are safe to run in CI without `OPENAI_API_KEY` / `CLAUDE_KEY` / `COINGLASS_API_KEY`.
- Whenever you add a new agent or change the aggregation formula in `compute_confidence_score` / `agent_for_twitter_analysis`, run the corresponding file — it covers the edge cases (abstain, zero mean, decay zone).
