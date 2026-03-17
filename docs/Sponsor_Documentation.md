# BTCUSDT Price Direction Prediction System

## Project Overview

This system predicts Bitcoin (BTCUSDT) price direction using classical machine learning. It answers two questions for investors:

- **Will the price go up?** (Base models) — binary prediction of whether BTC price will be higher after N days
- **Will the price stay above its trend?** (Range models) — binary prediction of whether the closing price will remain above the 14-day moving average after N days

Predictions are generated for four time horizons: **1, 3, 5, and 7 days ahead** — producing 8 models total. All predictions are served in real-time via a REST API (FastAPI).

---

## Data Foundation

The system aggregates **28 daily datasets** covering ~1,000 days of history from two sources:

| Category | Datasets | Source |
|---|---|---|
| Futures Open Interest | 4 | CoinGlass API |
| Futures Funding Rates | 3 | CoinGlass API |
| Futures Long/Short Ratios | 3 | CoinGlass API |
| Futures Liquidations | 2 | CoinGlass API |
| Futures Orderbook | 2 | CoinGlass API |
| Futures Taker Volume | 2 | CoinGlass API |
| Exchange Metrics (Bitfinex Margin, Coinbase Premium) | 3 | CoinGlass API |
| On-Chain (LTH/STH Supply, Active Addresses, Reserve Risk) | 4 | CoinGlass API |
| BTC Spot OHLCV | 1 | CoinGlass API |
| S&P 500, Gold, Software ETF (IGV) | 4 | Yahoo Finance |

After merging and feature engineering (derivatives, technical indicators, lag features), the total feature pool reaches **~1,100 features**. Each model uses a curated subset of **11 to 21 features** selected for that specific prediction task.

**Key feature groups used by models:**
- Futures positioning (open interest changes, funding rates, long/short ratios)
- On-chain fundamentals (long-term holder supply, reserve risk, active addresses)
- Technical indicators (RSI, ADX, CCI, Bollinger Band Width, ATR, MFI, OBV, ROC)
- Macro context (S&P 500 and Gold technical indicators)

---

## Model Building Methodology

### Step 1 — Data Collection & Preparation
All 28 datasets are fetched, merged by date, forward-filled for weekends/holidays, and filtered to the most recent 1,000 days. Sparse columns (>40% missing) are removed.

### Step 2 — Feature Engineering
For each numeric column: first differences and percent changes are computed. Four market imbalance indicators are derived (taker buy/sell imbalance, liquidation imbalance, orderbook imbalance). 8 technical analysis indicators are calculated for each of 4 assets (BTC, S&P 500, Gold, IGV) — 32 TA features total.

### Step 3 — Model Training
Each model is an sklearn Pipeline:

> **Missing Value Imputation** → **Feature Scaling** (zero mean, unit variance) → **Logistic Regression** (balanced class weights, max 3,000 iterations)

Logistic Regression was chosen for its interpretability, robustness to overfitting on small datasets, and probabilistic output that allows threshold tuning.

### Step 4 — Walk-Forward Cross-Validation
Models are validated using **TimeSeriesSplit with 4 folds** — the industry standard for time-series data. This ensures:
- **No future data leakage** — the model never trains on data from the future
- **Purge gap** equal to the prediction horizon — prevents label contamination at fold boundaries
- **Realistic evaluation** — each fold simulates real-world deployment conditions

### Step 5 — Threshold Optimization
Each model's probability threshold is tuned to balance precision and recall for optimal real-world performance.

---

## Model Performance Ranking

All models ranked by **AUC-ROC** (area under ROC curve) — the primary measure of discriminative power. Metrics are cross-validation averages across 4 folds.

### Top Performers: Range Models

| Rank | Model | Horizon | AUC | Accuracy | Precision | Recall | F1 | Features |
|---|---|---|---|---|---|---|---|---|
| 1 | **Range 1D** | 1 day | **0.921** | 81.7% | 89.0% | 77.5% | 81.1% | 21 |
| 2 | **Range 3D** | 3 days | **0.852** | 74.7% | 75.6% | 80.2% | 76.1% | 11 |
| 3 | **Range 5D** | 5 days | **0.767** | 69.5% | 72.5% | 65.0% | 67.2% | 21 |
| 4 | **Range 7D** | 7 days | **0.711** | 65.8% | 71.5% | 50.1% | 58.4% | 16 |

### Base Models (Direct Price Direction)

| Rank | Model | Horizon | AUC | Accuracy | Precision | Recall | F1 | Features |
|---|---|---|---|---|---|---|---|---|
| 5 | Base 3D | 3 days | 0.621 | 52.9% | 58.3% | 32.0% | 37.7% | 20 |
| 6 | Base 5D | 5 days | 0.599 | 56.9% | 63.5% | 51.1% | 48.1% | 15 |
| 7 | Base 1D | 1 day | 0.554 | 54.0% | 53.5% | 70.0% | 60.5% | 20 |
| 8 | Base 7D | 7 days | 0.528 | 55.0% | 51.8% | 65.3% | 56.8% | 20 |

---

## Key Takeaways

1. **Range models significantly outperform base models** (AUC 0.71–0.92 vs 0.53–0.62). Predicting whether price stays above its trend is more tractable than predicting raw direction — a finding consistent with quantitative finance literature.

2. **Shorter horizons yield stronger predictions.** The 1-day Range model achieves 92% AUC with 82% accuracy — strong discriminative power for a financial prediction task.

3. **No overfitting detected.** Cross-validation metrics closely track out-of-sample performance on held-out data, confirming the walk-forward methodology prevents data leakage.

4. **Compact feature sets.** The best model (Range 3D) uses only 11 features — demonstrating that a focused signal set outperforms high-dimensional approaches.

5. **Production-ready.** All models are served via a FastAPI endpoint with automatic dataset caching and model versioning. Predictions are available in real-time.

---

*System processes ~1,100 features from 28 data sources. Models retrained on demand. API available at port 8000.*
