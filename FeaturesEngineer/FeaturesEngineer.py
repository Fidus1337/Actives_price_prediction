import numpy as np
import pandas as pd
from functools import reduce

N_DAYS = 1

class FeaturesEngineer:
    
    # ---------- 1) Нормализация спот-колонок ----------
    def ensure_spot_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mapping = {
            "open": "spot_price_history__open",
            "high": "spot_price_history__high",
            "low": "spot_price_history__low",
            "close": "spot_price_history__close",
            "volume_usd": "spot_price_history__volume_usd",
        }
        # переименуем только если целевой префикс-колонки ещё нет
        rename = {}
        for old, new in mapping.items():
            if old in out.columns and new not in out.columns:
                rename[old] = new
        if rename:
            out = out.rename(columns=rename)
        return out
    
    
    # ---------- 2) Бинарный таргет на завтра ----------
    def add_y_up_1d(self, df: pd.DataFrame, close_col: str = "spot_price_history__close") -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date", kind="stable").reset_index(drop=True)
    
        c = pd.to_numeric(out[close_col], errors="coerce")
        out["y_up_1d"] = (c.shift(-1*N_DAYS) > c).astype("Int64")
        return out
        
    # ---------- 3) Бинарный таргет на предсказание кастомного количества дней ----------
    def add_y_up_custom(self, df: pd.DataFrame, horizon: int, close_col: str = "spot_price_history__close") -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date", kind="stable").reset_index(drop=True)
    
        target_column_name = f"y_up_{horizon}d"
        c = pd.to_numeric(out[close_col], errors="coerce")
        out[target_column_name] = (c.shift(-horizon) > c).astype("Int64")
        return out
    
    # ---------- 2.1) Бинарный таргет на N дней вперёд ----------
    def add_y_up_nd(self, df: pd.DataFrame, horizon: int = 1, close_col: str = "spot_price_history__close") -> pd.DataFrame:
        """
        Создаёт бинарный таргет: цена через `horizon` дней выше текущей (1) или нет (0).
        Колонка будет называться y_up_{horizon}d.
        """
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date", kind="stable").reset_index(drop=True)
    
        c = pd.to_numeric(out[close_col], errors="coerce")
        target_col = f"y_up_{horizon}d"
        out[target_col] = (c.shift(-horizon) > c).astype("Int64")
        return out
    
        # ---------- 3) Feature engineering: diff/pct + imbalance ----------
    def add_engineered_features(self, df: pd.DataFrame, horizon=1) -> pd.DataFrame:
        out = df.copy()
        eps = 1e-12
        target_column_name = f"y_up_{horizon}d"

        # diff/pct для всех numeric (кроме таргета)
        base_numeric = [
            c for c in out.columns
            if c not in {"date", target_column_name}
            and pd.api.types.is_numeric_dtype(out[c])
        ]
        for c in base_numeric:
            out[c + "__diff1"] = out[c].diff(1)
            out[c + "__pct1"] = out[c].pct_change(1)

        # imbalances (если пары колонок есть)
        def _imbalance(num_col_a, num_col_b, new_col):
            if num_col_a in out.columns and num_col_b in out.columns:
                a = pd.to_numeric(out[num_col_a], errors="coerce")
                b = pd.to_numeric(out[num_col_b], errors="coerce")
                out[new_col] = (a - b) / (a + b + eps)

        _imbalance(
            "futures_v2_taker_buy_sell_volume_history__taker_buy_volume_usd",
            "futures_v2_taker_buy_sell_volume_history__taker_sell_volume_usd",
            "feat__taker_imbalance_v2",
        )
        _imbalance(
            "futures_aggregated_taker_buy_sell_volume_history__aggregated_buy_volume_usd",
            "futures_aggregated_taker_buy_sell_volume_history__aggregated_sell_volume_usd",
            "feat__taker_imbalance_agg",
        )
        _imbalance(
            "futures_liquidation_history__short_liquidation_usd",
            "futures_liquidation_history__long_liquidation_usd",
            "feat__liq_imbalance_short_minus_long",
        )
        _imbalance(
            "futures_orderbook_ask_bids_history__bids_usd",
            "futures_orderbook_ask_bids_history__asks_usd",
            "feat__orderbook_imbalance_usd",
        )

        # почистим бесконечности
        out = out.replace([np.inf, -np.inf], np.nan)
        return out