"""
Чистые функции для кросс-источниковых фич `feat__*` и `cb_premium_*`.

Каждая функция принимает pd.Series (одну или несколько), не привязана к
конкретным именам колонок CoinGlass и не имеет состояния. Маппинг
"какая колонка из какого raw-датафрейма" решается уровнем выше
(см. get_dataset_by_all_features._resolve_col).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "funding_minus_oi_weight",
    "taker_imbalance",
    "liq_imbalance_short_minus_long",
    "liq_total_pct1",
    "premium_abs",
    "premium_rate_bps",
]


def funding_minus_oi_weight(
    funding: pd.Series,
    oi_weight_funding: pd.Series,
) -> pd.Series:
    """
    Спред между обычным funding и OI-взвешенным funding.

    >0 — биржи, не доминирующие по OI, держат более перегретый funding;
    <0 — крупные по OI биржи перегреты сильнее.
    """
    return funding.astype(float) - oi_weight_funding.astype(float)


def taker_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    eps: float = 1e-9,
) -> pd.Series:
    """
    Нормированный taker buy/sell imbalance: (buy - sell) / (buy + sell).
    Диапазон ~[-1, 1]. >0 — агрессивные покупки доминируют.
    """
    b = buy_volume.astype(float)
    s = sell_volume.astype(float)
    return (b - s) / (b + s + eps)


def liq_imbalance_short_minus_long(
    long_liq: pd.Series,
    short_liq: pd.Series,
    eps: float = 1e-9,
) -> pd.Series:
    """
    Нормированный дисбаланс ликвидаций: (short_liq - long_liq) / total.
    >0 — выносят шорты (бычье принуждение), <0 — выносят лонги.
    """
    l = long_liq.astype(float)
    s = short_liq.astype(float)
    total = l + s
    return (s - l) / (total + eps)


def liq_total_pct1(
    long_liq: pd.Series,
    short_liq: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """Суточное процентное изменение суммарных ликвидаций."""
    total = long_liq.astype(float) + short_liq.astype(float)
    return total.pct_change(periods=periods)


def premium_abs(premium: pd.Series) -> pd.Series:
    """Абсолютное значение Coinbase premium (амплитуда давления)."""
    return premium.astype(float).abs()


def premium_rate_bps(
    premium_rate: pd.Series,
    rate_is_percent: bool = True,
) -> pd.Series:
    """
    Coinbase premium rate в базисных пунктах.

    rate_is_percent=True  → rate уже в процентах (0.17 = 0.17%), bps = rate * 100.
    rate_is_percent=False → rate в долях (0.0017 = 0.17%), bps = rate * 10_000.

    CoinGlass v4 отдаёт rate в процентах — оставлено по умолчанию True.
    """
    factor = 100.0 if rate_is_percent else 10_000.0
    return premium_rate.astype(float) * factor
