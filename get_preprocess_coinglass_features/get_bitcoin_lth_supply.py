"""
Функции для получения Bitcoin Index метрик с CoinGlass API.
(LTH Supply, Active Addresses и др.)
"""
import pandas as pd
import numpy as np

from .helpers._coinglass_get_dataframe import _coinglass_get_dataframe
from .helpers._coinglass_normalize_time_to_date import _coinglass_normalize_time_to_date
from .helpers._prefix_columns import _prefix_columns


def get_bitcoin_lth_supply(
    api_key: str,
    pct_window: int = 30,
    z_window: int = 180,
    slope_window: int = 14,
    prefix: str = "index_btc_lth_supply",
) -> pd.DataFrame:
    """
    Bitcoin Long-Term Holder Supply с расчётными фичами для прогнозирования.
    
    Args:
        api_key: API ключ CoinGlass
        pct_window: Окно для процентного изменения (дней)
        z_window: Окно для z-score (дней)
        slope_window: Окно для slope/velocity (дней)
        prefix: Префикс для колонок
    
    Returns:
        DataFrame с колонками:
          - date
          - {prefix}__price
          - {prefix}__lth_supply
          - {prefix}__supply_pct{pct_window}
          - {prefix}__supply_z{z_window}
          - {prefix}__supply_slope{slope_window}
    """
    df = _coinglass_get_dataframe(
        endpoint="/index/bitcoin-long-term-holder-supply",
        api_key=api_key,
    )
    
    if df.empty:
        return df
    
    # timestamp -> date (этот эндпоинт использует timestamp, а не time)
    if "timestamp" in df.columns:
        df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
        df = df.drop(columns=["timestamp"])
    
    # Нормализация числовых колонок
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["lth_supply"] = pd.to_numeric(df.get("long_term_holder_supply"), errors="coerce")
    
    # Очистка и сортировка
    df = (
        df[["date", "price", "lth_supply"]]
        .dropna(subset=["date"])
        .sort_values("date", kind="stable")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    
    s = df["lth_supply"].astype(float)
    
    # Feature 1: pct change over N days (supply expansion/contraction proxy)
    df[f"supply_pct{pct_window}"] = s / s.shift(pct_window) - 1.0
    
    # Feature 2: rolling z-score (regime-normalized supply)
    minp = max(30, z_window // 3)
    roll = s.rolling(z_window, min_periods=minp)
    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0.0, np.nan)
    df[f"supply_z{z_window}"] = (s - mu) / sd
    
    # Feature 3: slope / velocity
    df[f"supply_slope{slope_window}"] = s.diff(slope_window) / float(slope_window)
    
    # Префикс
    df = _prefix_columns(df, prefix=prefix, keep=("date",))
    
    return df
