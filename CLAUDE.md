# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin price direction prediction system using CoinGlass API data. Trains Logistic Regression models to predict if BTC price will be higher after N days (1d, 3d, 7d horizons).

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run full pipeline (all configs from config.json)
python Main.py

# API key required in dev.env:
# COINGLASS_API_KEY=your_key_here
```

## Architecture

### Data Pipeline Flow
```
CoinGlass API -> FeaturesGetter -> merge_by_date -> FeaturesEngineer -> CorrelationsAnalyzer -> ML Training
```

### Key Modules

**FeaturesGetterModule/** - API data fetching
- `FeaturesGetter.py` - Main class wrapping CoinGlass API endpoints
- `features_endpoints.json` - Endpoint configurations (paths, default params)
- `helpers/` - Utility functions for API response processing

**FeaturesEngineer/** - Feature engineering
- `ensure_spot_prefix()` - Normalizes OHLCV column names
- `add_y_up_custom()` - Creates binary target (price up after N days)
- `add_engineered_features()` - Adds diff/pct changes and imbalance features

**CorrelationsAnalyzer/** - Statistical analysis
- `corr_report()` - Correlation with p-values and FDR correction
- `group_effect_report()` - Cohen's d effect sizes between y=0/y=1 groups

### Training Scripts

- `logistic_reg_model_train.py` - Walk-forward CV, hyperparameter tuning grid
- `quality_metrics.py` - ROC plots, metrics vs threshold analysis
- `get_features_from_API.py` - Orchestrates fetching all feature datasets

### Configuration

`config.json` defines experiment runs:
```json
{
  "runs": [
    {"name": "baseline_1d", "N_DAYS": 1, "base_feats": [...]}
  ]
}
```

## Data Conventions

- All DataFrames have `date` column (datetime)
- Feature columns use prefix pattern: `{source}__{metric}` (e.g., `futures_open_interest_history__close`)
- Derived features append suffix: `__diff1`, `__pct1`, `__lag{N}`
- Target column: `y_up_{N}d` (binary: 1 if price higher after N days)

## Key Patterns

- TimeSeriesSplit for walk-forward validation (no future leakage)
- Pipeline: SimpleImputer -> StandardScaler -> LogisticRegression
- Coverage filtering: features need >= 85% non-null values
- Graphics saved to `graphics/{config_name}/`
