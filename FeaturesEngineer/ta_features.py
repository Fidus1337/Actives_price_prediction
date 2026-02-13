import pandas as pd
import ta


def add_ta_features_for_asset(df: pd.DataFrame, prefix: str, volume_col_override: str | None = None) -> pd.DataFrame:
    """Добавляет TA-индикаторы для актива с заданным префиксом.

    Parameters:
        prefix: префикс колонок актива (e.g. "gold", "sp500", "spot_price_history")
        volume_col_override: полное имя volume-колонки, если оно не {prefix}__volume
                             (e.g. "spot_price_history__volume_usd" для BTC)
    """
    df = df.copy()

    required = ['open', 'close', 'high', 'low', 'volume']
    col_map = {col: f"{prefix}__{col}" for col in required}

    # Позволяем переопределить имя volume-колонки
    if volume_col_override:
        col_map['volume'] = volume_col_override

    missing = [col_map[c] for c in required if col_map[c] not in df.columns]
    if missing:
        print(f"  Пропущены колонки для {prefix}: {missing}")
        return df

    temp_df = pd.DataFrame({
        'open': df[col_map['open']].values,
        'high': df[col_map['high']].values,
        'low': df[col_map['low']].values,
        'close': df[col_map['close']].values,
        'volume': df[col_map['volume']].values
    })

    temp_with_ta = ta.add_all_ta_features(
        temp_df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=False
    )

    original_cols = {'open', 'high', 'low', 'close', 'volume'}
    ta_cols = [c for c in temp_with_ta.columns if c not in original_cols]

    for col in ta_cols:
        df.loc[df.index, f"{prefix}__{col}"] = temp_with_ta[col].values

    print(f"  Добавлено {len(ta_cols)} TA-фичей для {prefix}")
    return df
