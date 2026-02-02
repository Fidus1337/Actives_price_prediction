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
import json
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
from ModelsTrainer.logistic_reg_model_train import add_range_target


@dataclass
class PredictionResult:
    """Container for a single date's prediction results."""
    date: str
    prediction: int
    probability: float


class Predictor:
    """
    Makes predictions using a pre-trained model for a specific configuration.

    Example:
        predictor = Predictor("base_model_1d")
        results = predictor.predict_by_dates(["2025-01-20"])
    """

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

        # Determine model type from config name
        if "range" in config_name:
            self.model_type = "range"
        else:
            self.model_type = "base"

        # Load environment variables (API key)
        load_dotenv(env_path)
        self.api_key = os.getenv("COINGLASS_API_KEY")
        if not self.api_key:
            raise ValueError(f"COINGLASS_API_KEY not found in {env_path}")

        # Load configuration
        self.config = self._load_config()

        # Extract key parameters from config
        self.n_days = self.config["N_DAYS"]
        self.ma_window = self.config.get("ma_window", 14)
        self.base_feats = self.config["base_feats"]
        self.range_feats = self.config.get("range_feats", [])

        # Load model and model features
        self.model = self._load_model()
        self.model_features = self._load_model_features()

        # Initialize helpers
        self.getter = FeaturesGetter(api_key=self.api_key)
        self.features_engineer = FeaturesEngineer()

        # Cache for prepared data
        self._prepared_df: Optional[pd.DataFrame] = None

    def _load_config(self) -> dict:
        """Load and return the specific run config from config.json."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        runs = config.get("runs", [])
        for run in runs:
            if run.get("name") == self.config_name:
                return run

        available = [r.get("name") for r in runs]
        raise ValueError(
            f"Config '{self.config_name}' not found. Available: {available}"
        )

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

    def _fetch_and_prepare_data(self) -> pd.DataFrame:
        """Fetch fresh data from API and apply feature engineering."""
        print("Fetching features from API...")
        dfs = get_features(self.getter, self.api_key)
        df_all = merge_by_date(dfs, how="outer", dedupe="last")
        print(f"Features gathered. Shape: {df_all.shape}")

        # Normalize spot columns
        df0 = self.features_engineer.ensure_spot_prefix(df_all)

        # Add target column
        df1 = self.features_engineer.add_y_up_custom(
            df0,
            horizon=self.n_days,
            close_col="spot_price_history__close"
        )

        # Add engineered features
        df2 = self.features_engineer.add_engineered_features(df1, horizon=self.n_days)

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
        """Filter DataFrame to last n_dates EXCLUDING today."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").reset_index(drop=True)

        # Exclude today
        today = pd.Timestamp.now().normalize()
        df = df[df["date"] < today]

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
            print(f"Warning: Missing features: {missing}")

        X = df[available_feats]

        proba = self.model.predict_proba(X)[:, 1]
        pred = self.model.predict(X)

        results = []
        for i, date in enumerate(df["date"].values):
            results.append(PredictionResult(
                date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                prediction=int(pred[i]),
                probability=float(proba[i]),
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
        if self._prepared_df is None:
            self._prepared_df = self._fetch_and_prepare_data()

        df_pred = self._get_prediction_dates(self._prepared_df, n_dates)

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
        if self._prepared_df is None:
            self._prepared_df = self._fetch_and_prepare_data()

        df_pred = self._filter_by_dates(self._prepared_df, dates)

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
    predictor = Predictor("range_model_1d")
    results = predictor.predict_by_dates(["2025-02-01"])
    for r in results:
        print(f"{r.date}: pred={r.prediction} (proba={r.probability:.3f})")
