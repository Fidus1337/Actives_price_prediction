"""
Сборка широкого фичевого датасета на основе FeaturesGetterRework + feature_transforms
+ cross_features.

Шаги:
    1) Тянем все raw-эндпоинты, описанные в features_endpoints.json.
    2) Считаем производные:
        - 8 spot-деривативов (intraday range, RV3/7, sma_rel 7/14, RSI/ADX/BBW, vol pct1)
        - close__pct1 для трёх OI-эндпоинтов
    3) Мержим всё по date в один широкий DataFrame.
    4) Поверх мерджа считаем cross-source feat__* и cb_premium_*.
    5) Опционально применяем z-score с заданными окнами к выбранным колонкам.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .get_features import FeaturesGetterRework, CoinGlassError
    from . import feature_transforms as ft
    from . import cross_features as cf
except ImportError:
    from get_features import FeaturesGetterRework, CoinGlassError
    import feature_transforms as ft
    import cross_features as cf


_HERE = Path(__file__).resolve().parent

SPOT_PREFIX = "spot_price_history"

# Эндпоинты, для которых добавляем close__pct1.
OI_PREFIXES = (
    "futures_open_interest_aggregated_history",
    "futures_open_interest_aggregated_stablecoin_history",
    "futures_open_interest_aggregated_coin_margin_history",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Возвращает первое имя из `candidates`, которое реально есть в df.columns.
    Если ничего не нашли — None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _spot_derivatives(spot_df: pd.DataFrame, prefix: str = SPOT_PREFIX) -> pd.DataFrame:
    """8 производных фич поверх spot_price_history."""
    high = spot_df[f"{prefix}__high"]
    low = spot_df[f"{prefix}__low"]
    close = spot_df[f"{prefix}__close"]
    volume = spot_df[f"{prefix}__volume_usd"]

    out = spot_df[["date"]].copy()
    out[f"{prefix}__intraday_range_pct"] = ft.intraday_range_pct(high, low, close)
    out[f"{prefix}__realized_vol_3d"] = ft.realized_vol(close, window=3)
    out[f"{prefix}__realized_vol_7d"] = ft.realized_vol(close, window=7)
    out[f"{prefix}__close__sma7_rel"] = ft.sma_rel(close, window=7)
    out[f"{prefix}__close__sma14_rel"] = ft.sma_rel(close, window=14)
    out[f"{prefix}__ta_rsi"] = ft.rsi(close, window=14)
    out[f"{prefix}__ta_adx"] = ft.adx(high, low, close, window=14)
    out[f"{prefix}__ta_bbw"] = ft.bbw(close, window=20, num_std=2.0)
    out[f"{prefix}__volume_usd__pct1"] = ft.pct_change(volume, periods=1)
    return out


def _oi_pct1(oi_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """close__pct1 для одного OI-эндпоинта."""
    close_col = f"{prefix}__close"
    if close_col not in oi_df.columns:
        return oi_df[["date"]].copy()
    out = oi_df[["date"]].copy()
    out[f"{prefix}__close__pct1"] = ft.pct_change(oi_df[close_col], periods=1)
    return out


def _add_cross_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Считает feat__* и cb_premium_* поверх уже смерженного wide DataFrame.
    Имена колонок у CoinGlass могут отличаться — поэтому каждый источник
    ищется через список кандидатов.
    """
    df = merged.copy()

    # ---------- feat__funding_minus_oi_weight ----------
    funding_col = _resolve_col(df, ["futures_funding_rate_history__close"])
    oi_w_col = _resolve_col(df, ["futures_funding_rate_oi_weight_history__close"])
    if funding_col and oi_w_col:
        df["feat__funding_minus_oi_weight"] = cf.funding_minus_oi_weight(df[funding_col], df[oi_w_col])
    else:
        print(f"[cross] funding_minus_oi_weight: нет источников ({funding_col=}, {oi_w_col=})")

    # ---------- feat__taker_imbalance_agg ----------
    taker_pref = "futures_aggregated_taker_buy_sell_volume_history"
    buy_col = _resolve_col(
        df,
        [
            f"{taker_pref}__aggregated_buy_volume_usd",
            f"{taker_pref}__buy",
            f"{taker_pref}__aggregated_buy_volume",
            f"{taker_pref}__buy_volume_usd",
        ],
    )
    sell_col = _resolve_col(
        df,
        [
            f"{taker_pref}__aggregated_sell_volume_usd",
            f"{taker_pref}__sell",
            f"{taker_pref}__aggregated_sell_volume",
            f"{taker_pref}__sell_volume_usd",
        ],
    )
    if buy_col and sell_col:
        df["feat__taker_imbalance_agg"] = cf.taker_imbalance(df[buy_col], df[sell_col])
    else:
        print(f"[cross] taker_imbalance_agg: нет источников ({buy_col=}, {sell_col=})")

    # ---------- feat__liq_imbalance_short_minus_long + feat__liq_total_pct1 ----------
    liq_pref = "futures_liquidation_aggregated_history"
    long_col = _resolve_col(
        df,
        [
            f"{liq_pref}__aggregated_long_liquidation_usd",
            f"{liq_pref}__long_liquidation_usd",
            f"{liq_pref}__long_liquidation",
            f"{liq_pref}__long",
        ],
    )
    short_col = _resolve_col(
        df,
        [
            f"{liq_pref}__aggregated_short_liquidation_usd",
            f"{liq_pref}__short_liquidation_usd",
            f"{liq_pref}__short_liquidation",
            f"{liq_pref}__short",
        ],
    )
    if long_col and short_col:
        df["feat__liq_imbalance_short_minus_long"] = cf.liq_imbalance_short_minus_long(
            df[long_col], df[short_col]
        )
        df["feat__liq_total_pct1"] = cf.liq_total_pct1(df[long_col], df[short_col])
    else:
        print(f"[cross] liq_*: нет источников ({long_col=}, {short_col=})")

    # ---------- cb_premium_rate_bps + cb_premium_abs ----------
    prem_col = _resolve_col(df, ["cb_premium__premium"])
    rate_col = _resolve_col(df, ["cb_premium__premium_rate"])
    if prem_col:
        df["cb_premium_abs"] = cf.premium_abs(df[prem_col])
    else:
        print("[cross] cb_premium_abs: нет 'cb_premium__premium'")
    if rate_col:
        df["cb_premium_rate_bps"] = cf.premium_rate_bps(df[rate_col], rate_is_percent=True)
    else:
        print("[cross] cb_premium_rate_bps: нет 'cb_premium__premium_rate'")

    return df


def _apply_zscores(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    """Добавляет колонки <col>__z{window} для каждой пары (col, window)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            print(f"  [zscore] skip: '{col}' нет в датафрейме")
            continue
        for w in windows:
            out[f"{col}__z{w}"] = ft.zscore(out[col], window=int(w))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_by_all_features(
    api_key: str | None = None,
    endpoints_path: str | Path | None = None,
    env_path: str | Path | None = None,
    z_score_columns: Iterable[str] | None = None,
    z_score_windows: Iterable[int] = (14, 30, 180),
    extra_overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Собирает широкий датасет фич: raw + spot/oi-deriv + cross + z-scores.

    Args:
        api_key:           CoinGlass API key (если None — берётся из .env).
        endpoints_path:    путь до features_endpoints.json.
        env_path:          путь до .env с COINGLASS_API_KEY.
        z_score_columns:   итерируемое имён колонок (с префиксами!), к которым
                           надо применить z-score. None = z-scores не считаем.
        z_score_windows:   список окон для z-score.
        extra_overrides:   {feature_name: {param: value}} — кастомные параметры
                           поверх default_params.

    Returns:
        DataFrame: одна колонка `date` + все фичи с их префиксами.
    """
    getter = FeaturesGetterRework(
        api_key=api_key,
        endpoints_path=endpoints_path,
        env_path=env_path,
    )

    overrides = extra_overrides or {}
    raw_by_name: dict[str, pd.DataFrame] = {}

    for name in getter.list_features():
        try:
            df = getter.get_feature(name, **overrides.get(name, {}))
        except CoinGlassError as e:
            print(f"[{name}] CoinGlass error: {e}")
            continue

        if df.empty:
            print(f"[{name}] пусто, пропускаю")
            continue

        print(f"[{name}] {df.shape[0]} строк, {df.shape[1] - 1} колонок: {list(df.columns)[1:]}")
        raw_by_name[name] = df

    if not raw_by_name:
        return pd.DataFrame()

    frames_to_merge: list[pd.DataFrame] = list(raw_by_name.values())

    # --- spot derivatives ---
    if "spot_price_history" in raw_by_name:
        deriv = _spot_derivatives(raw_by_name["spot_price_history"])
        print(f"[derivatives:spot] +{deriv.shape[1] - 1} фич")
        frames_to_merge.append(deriv)

    # --- OI close__pct1 ---
    for oi_pref in OI_PREFIXES:
        if oi_pref in raw_by_name:
            oi_deriv = _oi_pct1(raw_by_name[oi_pref], oi_pref)
            if oi_deriv.shape[1] > 1:
                print(f"[derivatives:{oi_pref}] +1 фича")
                frames_to_merge.append(oi_deriv)

    # --- merge all ---
    merged = reduce(
        lambda a, b: pd.merge(a, b, on="date", how="outer"),
        frames_to_merge,
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    # --- cross-source features ---
    merged = _add_cross_features(merged)

    # --- z-scores ---
    if z_score_columns:
        merged = _apply_zscores(merged, list(z_score_columns), list(z_score_windows))

    print(f"\nИтоговый shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    Z_COLUMNS = [
        "spot_price_history__close",
        "spot_price_history__volume_usd__pct1",
        "futures_funding_rate_oi_weight_history__close",
        "feat__funding_minus_oi_weight",
        "feat__taker_imbalance_agg",
    ]

    df = get_dataset_by_all_features(
        env_path=_HERE.parent.parent / "dev.env",
        z_score_columns=Z_COLUMNS,
        z_score_windows=(14, 30, 180),
    )

    print("\nКолонки итогового датасета:")
    for c in df.columns:
        print(f"  - {c}")

    out_path = _HERE / "all_features_dataset_1d.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
