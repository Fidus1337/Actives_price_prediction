"""
Чистые функции трансформации фич.

Каждая функция:
    - принимает pd.Series (или несколько) с raw данными,
    - возвращает pd.Series той же длины с производной фичей,
    - не зависит от внешних либ (pandas/numpy only) и не имеет состояния.

Имя возвращаемой Series задаёт вызывающий код — функции сами имена не ставят.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "pct_change",
    "intraday_range_pct",
    "realized_vol",
    "sma_rel",
    "rsi",
    "adx",
    "bbw",
    "zscore",
]


def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    """Процентное изменение за `periods` шагов: s[t]/s[t-periods] - 1."""
    return s.astype(float).pct_change(periods=periods)


def intraday_range_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Размах свечи в долях close: (high - low) / close."""
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float).replace(0.0, np.nan)
    return (h - l) / c


def realized_vol(
    close: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 365,
) -> pd.Series:
    """
    Реализованная волатильность как std логдоходностей за `window` дней.

    Args:
        close: ряд цены закрытия.
        window: окно std.
        annualize: если True — умножаем на sqrt(periods_per_year).
        periods_per_year: 365 для крипты (24/7), 252 для акций.
    """
    log_ret = np.log(close.astype(float)).diff()
    minp = max(2, window // 2)
    vol = log_ret.rolling(window, min_periods=minp).std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def sma_rel(close: pd.Series, window: int) -> pd.Series:
    """Отклонение цены от SMA: close / SMA(close, window) - 1."""
    c = close.astype(float)
    minp = max(1, window // 2)
    sma = c.rolling(window, min_periods=minp).mean().replace(0.0, np.nan)
    return c / sma - 1.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder RSI (классическая реализация через экспоненциальное сглаживание
    с alpha = 1/window). Возвращает значения в диапазоне 0..100.
    """
    c = close.astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Wilder ADX (Average Directional Index). 0..100. Сила тренда без направления.
    """
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=h.index)
    minus_dm = pd.Series(minus_dm, index=h.index)

    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / window
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()


def bbw(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Bands Width: (upper - lower) / middle.
    upper = SMA + num_std*std,  lower = SMA - num_std*std.
    """
    c = close.astype(float)
    minp = max(2, window // 2)
    mid = c.rolling(window, min_periods=minp).mean()
    sd = c.rolling(window, min_periods=minp).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return (upper - lower) / mid.replace(0.0, np.nan)


def zscore(s: pd.Series, window: int) -> pd.Series:
    """
    Скользящий z-score: (s - rolling_mean) / rolling_std.
    Min_periods = max(2, window // 3) — чтобы не выдавать бредовые z в самом начале.
    """
    x = s.astype(float)
    minp = max(2, window // 3)
    roll = x.rolling(window, min_periods=minp)
    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0.0, np.nan)
    return (x - mu) / sd
