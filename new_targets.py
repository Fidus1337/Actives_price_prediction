import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

def make_up7d_target_ret_threshold(
    close: pd.Series,
    horizon: int = 7,
    ret_thr: float = 0.02,   # 2% за 7 дней
    use_log: bool = True,
) -> pd.Series:
    close = close.astype(float)
    if use_log:
        fwd = np.log(close.shift(-horizon) / close)
        thr = np.log(1.0 + ret_thr)
    else:
        fwd = close.shift(-horizon) / close - 1.0
        thr = ret_thr

    y = (fwd >= thr).astype(float)
    y.iloc[-horizon:] = np.nan
    return y


def make_up7d_target_vol_scaled(
    close: pd.Series,
    horizon: int = 7,
    vol_window: int = 30,
    k: float = 0.5,
) -> pd.Series:
    close = close.astype(float)
    ret = np.log(close).diff()
    vol = ret.rolling(vol_window).std(ddof=0)

    fwd = np.log(close.shift(-horizon) / close)
    thr = k * vol * np.sqrt(horizon)

    y = (fwd >= thr).astype(float)
    y.iloc[-horizon:] = np.nan
    return y


def make_triple_barrier_target(
    df_ohlc: pd.DataFrame,
    horizon: int = 7,
    up: float = 0.05,       # +3% barrier
    down: float = 0.02,     # -2% barrier
    neutral_to: str = "nan" # "nan" | "endclose"
) -> pd.Series:
    """
    Требует df_ohlc с колонками: high, low, close
    Возвращает y in {0,1} (или NaN для neutral, если neutral_to="nan")
    """
    df = df_ohlc.copy()
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)

    y = np.full(n, np.nan, dtype=float)

    for i in range(n - horizon):
        c0 = c[i]
        up_level = c0 * (1.0 + up)
        dn_level = c0 * (1.0 - down)

        first = None  # ("up"|"down", t)
        for k in range(1, horizon + 1):
            if np.isfinite(h[i + k]) and h[i + k] >= up_level:
                first = ("up", k)
                break
            if np.isfinite(l[i + k]) and l[i + k] <= dn_level:
                first = ("down", k)
                break

        if first is None:
            if neutral_to == "endclose":
                y[i] = 1.0 if c[i + horizon] > c0 else 0.0
            else:
                y[i] = np.nan
        else:
            y[i] = 1.0 if first[0] == "up" else 0.0

    return pd.Series(y, index=df.index).dropna()
