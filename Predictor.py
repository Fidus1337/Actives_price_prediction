"""
Predictor module for making predictions using pre-trained models.

Usage:
    # Last N dates
    predictor = Predictor("baseline_1d")
    results = predictor.predict(n_dates=10)
    predictor.save_predictions()

    # Specific dates
    predictor = Predictor("baseline_1d")
    results = predictor.predict_by_dates(["2024-01-15", "2024-01-16", "2024-01-17"])
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
    base_model_pred: int
    base_model_proba: float
    range_model_pred: int
    range_model_proba: float


class Predictor:
    """
    Makes predictions using pre-trained models for a specific configuration.

    Example:
        predictor = Predictor("baseline_1d")
        results = predictor.predict()
        predictor.save_predictions()
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
            config_name: Name of the configuration (e.g., "baseline_1d")
            config_path: Path to config.json
            env_path: Path to environment file with API key
        """
        self.config_name = config_name
        self.config_path = config_path
        self.env_path = env_path

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
        self.range_feats = self.config.get("range_feats", ["range_pct", f"range_pct_ma{self.ma_window}"])

        # Load models
        self.model_base, self.model_range = self._load_models()

        # Initialize helpers
        self.getter = FeaturesGetter(api_key=self.api_key)
        self.features_engineer = FeaturesEngineer()

        # Cache for prepared data
        self._prepared_df: Optional[pd.DataFrame] = None
        self._prepared_df_range: Optional[pd.DataFrame] = None

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

    def _load_models(self) -> tuple:
        """Load base and range models from Models/{config_name}/."""
        models_folder = os.path.join("Models", self.config_name)

        base_path = os.path.join(models_folder, f"model_base_{self.config_name}.joblib")
        range_path = os.path.join(models_folder, f"model_range_{self.config_name}.joblib")

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base model not found: {base_path}")
        if not os.path.exists(range_path):
            raise FileNotFoundError(f"Range model not found: {range_path}")

        model_base = joblib.load(base_path)
        model_range = joblib.load(range_path)

        print(f"Loaded models from {models_folder}")
        return model_base, model_range

    def _fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetch fresh data from API and apply feature engineering.
        Replicates the Main.py pipeline exactly.
        """
        print("Fetching features from API...")
        dfs = get_features(self.getter, self.api_key)
        df_all = merge_by_date(dfs, how="outer", dedupe="last")
        print(f"Features gathered. Shape: {df_all.shape}")

        # Normalize spot columns
        print("Normalizing spot columns...")
        df0 = self.features_engineer.ensure_spot_prefix(df_all)

        # Add target column (not used for prediction, but needed for consistency)
        print(f"Adding target column (horizon={self.n_days}d)...")
        df1 = self.features_engineer.add_y_up_custom(
            df0,
            horizon=self.n_days,
            close_col="spot_price_history__close"
        )

        # Add engineered features
        print("Adding engineered features...")
        df2 = self.features_engineer.add_engineered_features(df1, horizon=self.n_days)
        print(f"Feature engineering complete. Shape: {df2.shape}")

        return df2

    def _prepare_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add range target features required for range model."""
        HIGH_COL = "spot_price_history__high"
        LOW_COL = "spot_price_history__low"
        CLOSE_COL = "spot_price_history__close"

        df_range = add_range_target(
            df,
            high_col=HIGH_COL,
            low_col=LOW_COL,
            close_col=CLOSE_COL,
            ma_window=self.ma_window,
            horizon=self.n_days,
            use_pct=True,
            baseline_shift=1,
        )

        return df_range

    def _get_prediction_dates(self, df: pd.DataFrame, n_dates: int = 10) -> pd.DataFrame:
        """
        Filter DataFrame to last n_dates EXCLUDING today.

        Returns filtered DataFrame sorted by date (oldest to newest).
        """
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
        """
        Filter DataFrame to specific dates.

        Args:
            df: DataFrame with 'date' column
            dates: List of date strings (e.g., ["2024-01-15", "2024-01-16"])

        Returns:
            Filtered DataFrame sorted by date.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert input dates to timestamps
        target_dates = pd.to_datetime(dates, errors="coerce").dropna().tolist()

        if len(target_dates) == 0:
            raise ValueError("No valid dates provided")

        # Filter to requested dates
        mask = df["date"].isin(target_dates)
        df = df.loc[mask].sort_values("date", kind="stable").reset_index(drop=True)

        return df

    def predict(self, n_dates: int = 10) -> list[PredictionResult]:
        """
        Generate predictions for the last n_dates excluding today.

        Args:
            n_dates: Number of dates to predict (default: 10)

        Returns:
            List of PredictionResult objects with predictions from both models.
        """
        # Fetch and prepare data if not cached
        if self._prepared_df is None:
            self._prepared_df = self._fetch_and_prepare_data()
            self._prepared_df_range = self._prepare_range_features(self._prepared_df)

        # Get the dates for prediction
        df_pred = self._get_prediction_dates(self._prepared_df, n_dates)
        df_pred_range = self._get_prediction_dates(self._prepared_df_range, n_dates)

        if len(df_pred) == 0:
            print("Warning: No dates available for prediction")
            return []

        # Prepare features for base model
        base_feats_available = [c for c in self.base_feats if c in df_pred.columns]
        missing_base = set(self.base_feats) - set(base_feats_available)
        if missing_base:
            print(f"Warning: Missing base features: {missing_base}")

        X_base = df_pred[base_feats_available]

        # Prepare features for range model
        range_feat_set = [c for c in (self.base_feats + self.range_feats)
                         if c in df_pred_range.columns]
        X_range = df_pred_range[range_feat_set]

        # Generate predictions
        print(f"Generating predictions for {len(df_pred)} dates...")
        base_proba = self.model_base.predict_proba(X_base)[:, 1]
        base_pred = self.model_base.predict(X_base)

        range_proba = self.model_range.predict_proba(X_range)[:, 1]
        range_pred = self.model_range.predict(X_range)

        # Build results
        results = []
        dates = df_pred["date"].values

        for i, date in enumerate(dates):
            result = PredictionResult(
                date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                base_model_pred=int(base_pred[i]),
                base_model_proba=float(base_proba[i]),
                range_model_pred=int(range_pred[i]),
                range_model_proba=float(range_proba[i]),
            )
            results.append(result)

        print(f"Generated {len(results)} predictions")
        return results

    def predict_by_dates(self, dates: list[str]) -> list[PredictionResult]:
        """
        Generate predictions for specific dates.

        Args:
            dates: List of date strings (e.g., ["2024-01-15", "2024-01-16"])

        Returns:
            List of PredictionResult objects with predictions from both models.
            Only dates that exist in the data will be returned.
        """
        # Fetch and prepare data if not cached
        if self._prepared_df is None:
            self._prepared_df = self._fetch_and_prepare_data()
            self._prepared_df_range = self._prepare_range_features(self._prepared_df)

        assert self._prepared_df_range is not None

        # Filter to requested dates
        df_pred = self._filter_by_dates(self._prepared_df, dates)
        df_pred_range = self._filter_by_dates(self._prepared_df_range, dates)

        if len(df_pred) == 0:
            print(f"Warning: None of the requested dates found in data: {dates}")
            return []

        # Report which dates were found
        found_dates = df_pred["date"].dt.strftime("%Y-%m-%d").tolist()
        missing_dates = set(dates) - set(found_dates)
        if missing_dates:
            print(f"Warning: Dates not found in data: {missing_dates}")
        print(f"Found {len(found_dates)} dates: {found_dates}")

        # Prepare features for base model
        base_feats_available = [c for c in self.base_feats if c in df_pred.columns]
        missing_base = set(self.base_feats) - set(base_feats_available)
        if missing_base:
            print(f"Warning: Missing base features: {missing_base}")

        X_base = df_pred[base_feats_available]

        # Prepare features for range model
        range_feat_set = [c for c in (self.base_feats + self.range_feats)
                         if c in df_pred_range.columns]
        X_range = df_pred_range[range_feat_set]

        # Generate predictions
        print(f"Generating predictions for {len(df_pred)} dates...")
        base_proba = self.model_base.predict_proba(X_base)[:, 1]
        base_pred = self.model_base.predict(X_base)

        range_proba = self.model_range.predict_proba(X_range)[:, 1]
        range_pred = self.model_range.predict(X_range)

        # Build results
        results = []
        result_dates = df_pred["date"].values

        for i, date in enumerate(result_dates):
            result = PredictionResult(
                date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                base_model_pred=int(base_pred[i]),
                base_model_proba=float(base_proba[i]),
                range_model_pred=int(range_pred[i]),
                range_model_proba=float(range_proba[i]),
            )
            results.append(result)

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

        # Convert to serializable format
        output_data = {
            "config_name": self.config_name,
            "n_days_horizon": self.n_days,
            "ma_window": self.ma_window,
            "generated_at": datetime.now().isoformat(),
            "predictions": [
                {
                    "date": p.date,
                    "base_model": {
                        "prediction": p.base_model_pred,
                        "probability": round(p.base_model_proba, 6),
                    },
                    "range_model": {
                        "prediction": p.range_model_pred,
                        "probability": round(p.range_model_proba, 6),
                    },
                }
                for p in predictions
            ],
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {output_path}")
        return output_path


def run_predictor(
    config_name: Optional[str] = None,
    n_dates: int = 10,
    dates: Optional[list[str]] = None,
    save_locally: bool = False
):
    """
    Run predictor for one or all configs.

    Args:
        config_name: Config name or None for all configs
        n_dates: Number of dates to predict (ignored if dates is provided)
        dates: List of specific dates to predict (e.g., ["2024-01-15", "2024-01-16"])
    """
    if config_name:
        # Single config
        predictor = Predictor(config_name)
        if dates:
            results = predictor.predict_by_dates(dates)
        else:
            results = predictor.predict(n_dates=n_dates)
        output_path = predictor.save_predictions(results)
        print(f"[{config_name}] Saved {len(results)} predictions to: {output_path}")
    else:
        # All configs
        with open("config.json", "r", encoding="utf-8") as f:
            configs = json.load(f).get("runs", [])

        print(f"Running predictions for {len(configs)} configurations")
        print("=" * 60)

        for cfg in configs:
            name = cfg.get("name")
            print(f"\n--- Processing: {name} ---")
            try:
                predictor = Predictor(name)
                if dates:
                    results = predictor.predict_by_dates(dates)
                else:
                    results = predictor.predict(n_dates=n_dates)
                    
                if save_locally:
                    output_path = predictor.save_predictions(results)
                    print(f"[{name}] SUCCESS: {len(results)} predictions saved")
                    
            except Exception as e:
                print(f"[{name}] ERROR: {e}")

        print("\n" + "=" * 60)
        print("All predictions complete")

if __name__ == "__main__":
    predictor = Predictor("baseline_1d")

    # По конкретным датам
    results = predictor.predict_by_dates(["2025-01-20", "2025-01-21", "2025-01-22"])
    for r in results:
        print(f"{r.date}: base={r.base_model_pred} ({r.base_model_proba:.3f}), "
              f"range={r.range_model_pred} ({r.range_model_proba:.3f})")
