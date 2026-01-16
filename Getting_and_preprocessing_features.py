"""
Модуль для получения и предобработки фичей с CoinGlass API.
"""
import pandas as pd
from get_preprocess_coinglass_features.get_coinglass_history import get_coinglass_history
from get_preprocess_coinglass_features.get_bitcoin_lth_supply import get_bitcoin_lth_supply


def fetch_all_coinglass_features(api_key: str) -> dict[str, pd.DataFrame]:
    """
    Загружает все фичи с CoinGlass API.
    
    Args:
        api_key: API ключ CoinGlass
    
    Returns:
        Словарь с DataFrame для каждой фичи:
          - open_interest_history
          - open_interest_aggregated
          - open_interest_stablecoin
          - open_interest_coin_margin
          - funding_rate_history
          - funding_rate_oi_weight
          - funding_rate_vol_weight
          - global_long_short_account_ratio
          - top_long_short_account_ratio
          - net_position
          - liquidation_history
          - liquidation_aggregated
          - orderbook_ask_bids
          - orderbook_aggregated
          - taker_buy_sell_volume
          - bitcoin_lth_supply
    """
    features = {}
    
    # Open Interest History
    features["open_interest_history"] = get_coinglass_history(
        endpoint_name="open_interest_history",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Open Interest Aggregated
    features["open_interest_aggregated"] = get_coinglass_history(
        endpoint_name="open_interest_aggregated",
        api_key=api_key,
        symbol="BTC",
        interval="1d",
    )
    
    # Open Interest Stablecoin
    features["open_interest_stablecoin"] = get_coinglass_history(
        endpoint_name="open_interest_stablecoin",
        api_key=api_key,
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Open Interest Coin Margin
    features["open_interest_coin_margin"] = get_coinglass_history(
        endpoint_name="open_interest_coin_margin",
        api_key=api_key,
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Funding Rate History
    features["funding_rate_history"] = get_coinglass_history(
        endpoint_name="funding_rate_history",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Funding Rate OI-Weighted
    features["funding_rate_oi_weight"] = get_coinglass_history(
        endpoint_name="funding_rate_oi_weight",
        api_key=api_key,
        symbol="BTC",
        interval="1d",
    )
    
    # Funding Rate Volume-Weighted
    features["funding_rate_vol_weight"] = get_coinglass_history(
        endpoint_name="funding_rate_vol_weight",
        api_key=api_key,
        symbol="BTC",
        interval="1d",
    )
    
    # Global Long/Short Account Ratio
    features["global_long_short_account_ratio"] = get_coinglass_history(
        endpoint_name="global_long_short_account_ratio",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Top Traders Long/Short Account Ratio
    features["top_long_short_account_ratio"] = get_coinglass_history(
        endpoint_name="top_long_short_account_ratio",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Net Position History
    features["net_position"] = get_coinglass_history(
        endpoint_name="net_position",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Liquidation History
    features["liquidation_history"] = get_coinglass_history(
        endpoint_name="liquidation_history",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Liquidation Aggregated
    features["liquidation_aggregated"] = get_coinglass_history(
        endpoint_name="liquidation_aggregated",
        api_key=api_key,
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Orderbook Ask/Bids History
    features["orderbook_ask_bids"] = get_coinglass_history(
        endpoint_name="orderbook_ask_bids",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Orderbook Aggregated
    features["orderbook_aggregated"] = get_coinglass_history(
        endpoint_name="orderbook_aggregated",
        api_key=api_key,
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Taker Buy/Sell Volume
    features["taker_buy_sell_volume"] = get_coinglass_history(
        endpoint_name="taker_buy_sell_volume",
        api_key=api_key,
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Bitcoin Long-Term Holder Supply (Index)
    features["bitcoin_lth_supply"] = get_bitcoin_lth_supply(
        api_key=api_key,
        pct_window=30,
        z_window=180,
        slope_window=14,
    )
    
    return features


def merge_all_features(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Объединяет все фичи в один DataFrame по колонке date.
    
    Args:
        features: Словарь с DataFrame (результат fetch_all_coinglass_features)
    
    Returns:
        Объединённый DataFrame со всеми фичами
    """
    dfs = list(features.values())
    
    if not dfs:
        return pd.DataFrame()
    
    # Начинаем с первого DataFrame
    result = dfs[0]
    
    # Последовательно объединяем остальные
    for df in dfs[1:]:
        if not df.empty and "date" in df.columns:
            result = result.merge(df, on="date", how="outer")
    
    # Сортируем по дате
    result = result.sort_values("date").reset_index(drop=True)
    
    return result


def print_features_summary(features: dict[str, pd.DataFrame]) -> None:
    """
    Выводит сводку по загруженным фичам.
    """
    print("=" * 70)
    print("СВОДКА ПО ЗАГРУЖЕННЫМ ФИЧАМ")
    print("=" * 70)
    
    for name, df in features.items():
        if df.empty:
            print(f"  {name}: ПУСТО")
        else:
            date_range = f"{df['date'].min()} — {df['date'].max()}" if "date" in df.columns else "N/A"
            print(f"  {name}: {len(df)} записей, {len(df.columns)} колонок, {date_range}")
    
    print("=" * 70)


# ============================================================================
# Пример использования
# ============================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv("dev.env")
    API_KEY = os.getenv("COINGLASS_API_KEY")
    
    if not API_KEY:
        raise ValueError("COINGLASS_API_KEY not found in dev.env")
    
    # Загружаем все фичи
    print("Загрузка фичей с CoinGlass...")
    features = fetch_all_coinglass_features(API_KEY)
    
    # Выводим сводку
    print_features_summary(features)
    
    # Объединяем в один DataFrame
    print("\nОбъединение фичей...")
    df_merged = merge_all_features(features)
    print(f"Итоговый DataFrame: {len(df_merged)} записей, {len(df_merged.columns)} колонок")
    print(f"Диапазон дат: {df_merged['date'].min()} — {df_merged['date'].max()}")
    print()
    print(df_merged.tail())
