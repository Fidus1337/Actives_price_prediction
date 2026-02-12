"""
Shared base data cache for Predictor instances.

Caches the result of the expensive shared pipeline steps
(API fetch + merge + ffill + engineered features + TA + lags)
that are identical for ALL models regardless of horizon or type.

Each Predictor then applies only model-specific target engineering
on top of a copy of this shared DataFrame.
"""

import threading
import time
from typing import Optional

import pandas as pd

from FeaturesGetterModule.FeaturesGetter import FeaturesGetter
from get_features_from_API import get_features
from FeaturesGetterModule.helpers._merge_features_by_date import merge_by_date
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer
from ModelsTrainer.logistic_reg_model_train import add_lags
from Models_builder_pipeline import add_ta_features_for_asset


class SharedBaseDataCache:
    """
    Thread-safe cache for the base DataFrame shared across all Predictor instances.

    The base DataFrame includes all pipeline steps up to (and including) add_lags(),
    but EXCLUDES model-specific target columns (y_up_Nd, range targets).
    """

    def __init__(self, api_key: str, ttl_seconds: float = 3600.0):
        self._api_key = api_key
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._base_df: Optional[pd.DataFrame] = None
        self._fetched_at: float = 0.0
        self._getter = FeaturesGetter(api_key=api_key)
        self._features_engineer = FeaturesEngineer()

    @property
    def is_stale(self) -> bool:
        if self._base_df is None:
            return True
        return (time.time() - self._fetched_at) > self._ttl_seconds

    def get_base_df(self) -> pd.DataFrame:
        """
        Return a COPY of the cached base DataFrame, refreshing if stale.

        Always returns a copy so callers can mutate freely
        (e.g., adding target columns) without corrupting the shared cache.
        """
        if self.is_stale:
            with self._lock:
                # Double-check after acquiring lock
                if self.is_stale:
                    self._base_df = self._fetch_base_data()
                    self._fetched_at = time.time()
        return self._base_df.copy()

    def clear(self) -> None:
        """Clear the cached base DataFrame (e.g., after retraining)."""
        with self._lock:
            self._base_df = None
            self._fetched_at = 0.0

    def refresh(self) -> None:
        """Force-refresh base data from API (called on every /api/predictions)."""
        print("SharedBaseDataCache: Refreshing data...")
        with self._lock:
            self._base_df = self._fetch_base_data()
            self._fetched_at = time.time()
        print(f"SharedBaseDataCache: Refreshed. Shape: {self._base_df.shape}")

    def _fetch_base_data(self) -> pd.DataFrame:
        """
        Execute shared pipeline steps 1-8 (identical for all models).

        1. get_features()                    -> 27 API calls
        2. merge_by_date()                   -> single DataFrame
        3. sort_values("date")
        4. ensure_spot_prefix()
        5. ffill()
        6. add_engineered_features()         -> diff1, pct1, imbalances
        7. add_ta_features_for_asset() x3    -> TA indicators
        8. add_lags()                        -> external market lags

        NOTE on step 6: horizon param is used only to exclude a target column
        from diff/pct computation, but that column doesn't exist yet at this
        point, so the output is identical regardless of horizon value.
        """
        print("SharedBaseDataCache: Fetching features from API...")
        dfs = get_features(self._getter, self._api_key)
        df_all = merge_by_date(dfs, how="outer", dedupe="last")
        df_all = df_all.sort_values("date").reset_index(drop=True)
        print(f"SharedBaseDataCache: Features gathered. Shape: {df_all.shape}")

        # Step 4: Normalize spot columns
        df0 = self._features_engineer.ensure_spot_prefix(df_all)

        # Step 5: Forward-fill NaN gaps
        feature_cols = [c for c in df0.columns if c != "date"]
        df0[feature_cols] = df0[feature_cols].ffill()

        # Step 6: Engineered features (horizon irrelevant, see docstring)
        df2 = self._features_engineer.add_engineered_features(df0)

        # Step 7: TA indicators
        df2 = add_ta_features_for_asset(df2, prefix="gold")
        df2 = add_ta_features_for_asset(df2, prefix="sp500")
        df2 = add_ta_features_for_asset(
            df2, prefix="spot_price_history",
            volume_col_override="spot_price_history__volume_usd"
        )

        # Step 8: Lag features for external market data
        EXTERNAL_LAGS = (1, 3, 5, 7, 10, 15)
        gold_cols = [c for c in df2.columns if c.startswith("gold__") and "__lag" not in c]
        sp500_cols = [c for c in df2.columns if c.startswith("sp500__") and "__lag" not in c]
        external_market_cols = gold_cols + sp500_cols

        if external_market_cols:
            print(f"SharedBaseDataCache: Adding lags for {len(external_market_cols)} external columns...")
            df2 = add_lags(df2, cols=external_market_cols, lags=EXTERNAL_LAGS)

        print(f"SharedBaseDataCache: Base data ready. Shape: {df2.shape}")
        return df2
