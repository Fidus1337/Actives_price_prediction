"""
Predictor module for making predictions using pre-trained models.

Usage:
    # Last N dates
    predictor = Predictor("base_model_1d")
    results = predictor.predict(n_dates=10)

    # Specific dates
    predictor = Predictor("range_model_1d")
    results = predictor.predict_by_dates(["2024-01-15", "2024-01-16"])
    predictor.save_predictions(results)
"""

import os
import re
import json
import time
import joblib
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from ModelsTrainer.logistic_reg_model_train import add_range_target, add_lags
from Models_builder_pipeline import add_ta_features_for_asset


@dataclass
class PredictionResult:
    """Container for a single date's prediction results."""
    date: str
    prediction: int
    probability: float
    spot_price: float | None = None


class Predictor:
    """
    Makes predictions using a pre-trained model for a specific configuration.

    Example:
        predictor = Predictor("base_model_1d")
        results = predictor.predict_by_dates(["2025-01-20"])
    """

    # Class-level shared cache (set by API startup, None for standalone use)
    _shared_cache = None  # type: Optional["SharedBaseDataCache"]

    @classmethod
    def set_shared_cache(cls, cache) -> None:
        """Attach a shared base data cache for all Predictor instances."""
        cls._shared_cache = cache

    @classmethod
    def clear_shared_cache(cls) -> None:
        """Clear the shared base data cache (e.g., after retraining)."""
        if cls._shared_cache is not None:
            cls._shared_cache.clear()

    def invalidate_cache(self) -> None:
        """Reset per-instance prepared DataFrame cache so it rebuilds from shared cache."""
        self._prepared_df = None
        self._prepared_at = 0.0

    def __init__(
        self,
        config_name: str,
        config_path: str = "config.json",
        env_path: str = "dev.env",
    ):
        """
        Initialize Predictor with configuration name.

        Args:
            config_name: Name of the configuration (e.g., "base_model_1d", "range_model_1d")
            config_path: Path to config.json
            env_path: Path to environment file with API key
        """
        self.config_name = config_name
        self.config_path = config_path
        self.env_path = env_path

        # Parse model type and horizon from name (always works)
        self.model_type, self.n_days = self._parse_name()
        self.ma_window = 14

        # Load environment variables (API key)
        load_dotenv(env_path)
        self.api_key = os.getenv("COINGLASS_API_KEY")
        if not self.api_key:
            raise ValueError(f"COINGLASS_API_KEY not found in {env_path}")

        # Config is optional — override defaults if found
        self.config = self._load_config_optional()
        if self.config:
            self.n_days = self.config["N_DAYS"]
            self.ma_window = self.config.get("ma_window", 14)
            self.base_feats = self.config["base_feats"]
            self.range_feats = self.config.get("range_feats", [])
        else:
            self.base_feats = []
            self.range_feats = []

        # Load model and model features
        self.model = self._load_model()
        self.model_features = self._load_model_features()

        # Initialize helpers
        self.getter = FeaturesGetter(api_key=self.api_key)
        self.features_engineer = FeaturesEngineer()

        # Cache for prepared data (with TTL)
        self._prepared_df: Optional[pd.DataFrame] = None
        self._prepared_at: float = 0.0
        self.cache_ttl_seconds: float = 3600  # 1 hour

    def _parse_name(self) -> tuple[str, int]:
        """Extract model_type and n_days from config_name like 'base_model_1d'."""
        model_type = "range" if "range" in self.config_name else "base"
        match = re.search(r"(\d+)d$", self.config_name)
        if not match:
            raise ValueError(
                f"Cannot parse N_DAYS from name '{self.config_name}'. "
                f"Expected pattern like 'base_model_1d'"
            )
        return model_type, int(match.group(1))

    def _load_config_optional(self) -> dict | None:
        """Try to load run config from config.json. Returns None if not found."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for run in config.get("runs", []):
                if run.get("name") == self.config_name:
                    return run
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return None

    def _load_model(self):
        """Load model from Models/{config_name}/."""
        models_folder = os.path.join("Models", self.config_name)
        model_path = os.path.join(models_folder, f"model_{self.model_type}_{self.config_name}.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Loaded {self.model_type} model from {model_path}")
        return model

    def _load_model_features(self) -> list[str]:
        """Load feature list from metrics JSON or model's feature_names_in_."""
        # Try to get features from metrics JSON first
        models_folder = os.path.join("Models", self.config_name)
        metrics_path = os.path.join(
            models_folder, f"metrics_{self.model_type}_{self.config_name}.json"
        )

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            features = metrics.get("features", [])
            if features:
                print(f"Loaded {len(features)} features from {metrics_path}")
                return features

        # Fallback: get feature names from sklearn model
        if hasattr(self.model, "feature_names_in_"):
            features = list(self.model.feature_names_in_)
            print(f"Loaded {len(features)} features from model.feature_names_in_")
            return features

        print("Warning: Could not determine features, falling back to config")
        return []

    def _is_cache_stale(self) -> bool:
        """Check if cached data has expired."""
        if self._prepared_df is None:
            return True
        return (time.time() - self._prepared_at) > self.cache_ttl_seconds

    def _get_prepared_df(self) -> pd.DataFrame:
        """Get prepared DataFrame, refreshing cache if stale."""
        if self._is_cache_stale():
            if self._prepared_df is not None:
                print("Cache expired, refreshing data...")
            self._prepared_df = self._fetch_and_prepare_data()
            self._prepared_at = time.time()
        return self._prepared_df

    def _fetch_and_prepare_data(self) -> pd.DataFrame:
        """Fetch data and apply feature engineering.

        If a shared cache is available (API mode), uses cached base DataFrame
        and applies only model-specific steps. Otherwise (standalone mode),
        performs the full pipeline.
        """
        if self._shared_cache is not None:
            return self._build_from_shared_cache()
        return self._fetch_full_pipeline()

    def _build_from_shared_cache(self) -> pd.DataFrame:
        """Build model-specific DataFrame from shared base data.

        Gets a copy of the shared base DataFrame (steps 1-8 already done)
        and applies only model-specific target columns (steps 9-10).
        """
        print(f"Using shared base data cache for {self.config_name}...")
        df2 = self._shared_cache.get_base_df()

        # Step 9: Add target column
        df2 = self.features_engineer.add_y_up_custom(
            df2,
            horizon=self.n_days,
            close_col="spot_price_history__close"
        )

        # Step 10: Add range features if needed
        if self.model_type == "range":
            df2 = add_range_target(
                df2,
                high_col="spot_price_history__high",
                low_col="spot_price_history__low",
                close_col="spot_price_history__close",
                ma_window=self.ma_window,
                horizon=self.n_days,
                use_pct=True,
                baseline_shift=1,
            )

        print(f"Feature engineering complete (from cache). Shape: {df2.shape}")
        return df2

    def _fetch_full_pipeline(self) -> pd.DataFrame:
        """Full pipeline for standalone use (no shared cache).

        Mirrors the training pipeline order from Models_builder_pipeline.main_pipeline():
        ensure_spot_prefix -> ffill -> add_engineered_features -> add_ta_features -> add_lags -> add_y_up_custom
        """
        print("Fetching features from API...")
        dfs = get_features(self.getter, self.api_key)
        df_all = merge_by_date(dfs, how="outer", dedupe="last")
        df_all = df_all.sort_values("date").reset_index(drop=True)
        print(f"Features gathered. Shape: {df_all.shape}")

        # Normalize spot columns
        df0 = self.features_engineer.ensure_spot_prefix(df_all)

        # Forward-fill NaN gaps (same as training pipeline)
        feature_cols = [c for c in df0.columns if c != "date"]
        df0[feature_cols] = df0[feature_cols].ffill()

        # Add engineered features (before target, same as training)
        df2 = self.features_engineer.add_engineered_features(df0, horizon=self.n_days)

        # Add TA indicators (same as training pipeline)
        df2 = add_ta_features_for_asset(df2, prefix="gold")
        df2 = add_ta_features_for_asset(df2, prefix="sp500")
        df2 = add_ta_features_for_asset(df2, prefix="spot_price_history",
                                         volume_col_override="spot_price_history__volume_usd")

        # Add lag features for external market data (same as training pipeline)
        EXTERNAL_LAGS = (1, 3, 5, 7, 10, 15)
        gold_cols = [c for c in df2.columns if c.startswith("gold__") and "__lag" not in c]
        sp500_cols = [c for c in df2.columns if c.startswith("sp500__") and "__lag" not in c]
        external_market_cols = gold_cols + sp500_cols

        if external_market_cols:
            print(f"Adding lag features for Gold ({len(gold_cols)}) and S&P500 ({len(sp500_cols)}) columns...")
            df2 = add_lags(df2, cols=external_market_cols, lags=EXTERNAL_LAGS)

        # Add target column (after engineered features + lags, same as training)
        df2 = self.features_engineer.add_y_up_custom(
            df2,
            horizon=self.n_days,
            close_col="spot_price_history__close"
        )

        # Add range features if needed
        if self.model_type == "range":
            df2 = add_range_target(
                df2,
                high_col="spot_price_history__high",
                low_col="spot_price_history__low",
                close_col="spot_price_history__close",
                ma_window=self.ma_window,
                horizon=self.n_days,
                use_pct=True,
                baseline_shift=1,
            )

        print(f"Feature engineering complete. Shape: {df2.shape}")
        return df2

    def _get_prediction_dates(self, df: pd.DataFrame, n_dates: int = 10) -> pd.DataFrame:
        """Filter DataFrame to last n_dates."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

        # Take last n_dates
        if len(df) > n_dates:
            df = df.tail(n_dates)

        return df.reset_index(drop=True)

    def _filter_by_dates(self, df: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
        """Filter DataFrame to specific dates."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        target_dates = pd.to_datetime(dates, errors="coerce").dropna().tolist()
        if len(target_dates) == 0:
            raise ValueError("No valid dates provided")

        mask = df["date"].isin(target_dates)
        df = df.loc[mask].sort_values("date", kind="stable").reset_index(drop=True)
        return df

    def _get_features(self) -> list[str]:
        """Get feature list from saved metrics (preferred) or config (fallback)."""
        # Prefer features saved during training (avoids mismatch)
        if self.model_features:
            return self.model_features

        # Fallback to config features (for backward compatibility)
        print("Warning: Using features from config (may cause mismatch)")
        if self.model_type == "range":
            return self.base_feats + self.range_feats
        return self.base_feats

    def _predict_df(self, df: pd.DataFrame) -> list[PredictionResult]:
        """Generate predictions for a prepared DataFrame."""
        features = self._get_features()
        available_feats = [c for c in features if c in df.columns]

        missing = set(features) - set(available_feats)
        if missing:
            raise ValueError(
                f"Missing {len(missing)} features required by model: {missing}. "
                f"Available: {len(available_feats)}/{len(features)}. "
                f"Check if CoinGlass API returned all required data."
            )

        X = df[available_feats]

        proba = self.model.predict_proba(X)[:, 1]
        pred = self.model.predict(X)

        has_spot = "spot_price_history__close" in df.columns
        results = []
        for i, date in enumerate(df["date"].values):
            spot_price = float(df["spot_price_history__close"].iloc[i]) if has_spot else None
            results.append(PredictionResult(
                date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                prediction=int(pred[i]),
                probability=float(proba[i]),
                spot_price=spot_price,
            ))

        return results

    def predict(self, n_dates: int = 10) -> list[PredictionResult]:
        """
        Generate predictions for the last n_dates excluding today.

        Args:
            n_dates: Number of dates to predict (default: 10)

        Returns:
            List of PredictionResult objects.
        """
        df = self._get_prepared_df()
        df_pred = self._get_prediction_dates(df, n_dates)

        if len(df_pred) == 0:
            print("Warning: No dates available for prediction")
            return []

        print(f"Generating predictions for {len(df_pred)} dates...")
        results = self._predict_df(df_pred)
        print(f"Generated {len(results)} predictions")
        return results

    def predict_by_dates(self, dates: list[str]) -> list[PredictionResult]:
        """
        Generate predictions for specific dates.

        Args:
            dates: List of date strings (e.g., ["2024-01-15", "2024-01-16"])

        Returns:
            List of PredictionResult objects. Only dates that exist in the data will be returned.
        """
        df = self._get_prepared_df()
        df_pred = self._filter_by_dates(df, dates)

        if len(df_pred) == 0:
            print(f"Warning: None of the requested dates found in data: {dates}")
            return []

        found_dates = df_pred["date"].dt.strftime("%Y-%m-%d").tolist()
        missing_dates = set(dates) - set(found_dates)
        if missing_dates:
            print(f"Warning: Dates not found in data: {missing_dates}")
        print(f"Found {len(found_dates)} dates: {found_dates}")

        print(f"Generating predictions for {len(df_pred)} dates...")
        results = self._predict_df(df_pred)
        print(f"Generated {len(results)} predictions")
        return results

    def save_predictions(
        self,
        predictions: Optional[list[PredictionResult]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save predictions to JSON file.

        Args:
            predictions: List of predictions (or None to run predict() first)
            output_path: Custom output path (default: Models/{config_name}/predictions.json)

        Returns:
            Path to saved JSON file.
        """
        if predictions is None:
            predictions = self.predict()

        if output_path is None:
            output_path = os.path.join("Models", self.config_name, "predictions.json")

        output_data = {
            "config_name": self.config_name,
            "model_type": self.model_type,
            "n_days_horizon": self.n_days,
            "generated_at": datetime.now().isoformat(),
            "predictions": [
                {
                    "date": p.date,
                    "prediction": p.prediction,
                    "probability": round(p.probability, 6),
                }
                for p in predictions
            ],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    predictor = Predictor("range_model_7d")
    results = predictor.predict_by_dates(["2025-02-01"])
    for r in results:
        print(f"{r.date}: pred={r.prediction} (proba={r.probability:.3f})")
