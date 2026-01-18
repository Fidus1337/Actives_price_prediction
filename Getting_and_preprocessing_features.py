"""
Модуль для получения и предобработки фичей с CoinGlass API.
"""
import pandas as pd
from FeaturesGetterModule.FeaturesGetter import FeaturesGetter


def fetch_all_coinglass_features(getter: FeaturesGetter) -> dict[str, pd.DataFrame]:
    """
    Загружает все фичи с CoinGlass API.
    
    Args:
        getter: Экземпляр FeaturesGetter с API ключом
    
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
          - top_long_short_position_ratio
          - net_position
          - liquidation_history
          - liquidation_aggregated
          - orderbook_ask_bids
          - orderbook_aggregated
          - taker_buy_sell_volume
          - taker_buy_sell_volume_aggregated
          - bitcoin_lth_supply
          - bitcoin_active_addresses
          - bitcoin_sth_supply
          - bitcoin_reserve_risk
          - bitfinex_margin_long_short
          - coinbase_premium_index
          - cgdi_index
    """
    features = {}
    
    # Open Interest History
    features["open_interest_history"] = getter.get_history(
        endpoint_name="open_interest_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Open Interest Aggregated
    features["open_interest_aggregated"] = getter.get_history(
        endpoint_name="open_interest_aggregated",
        symbol="BTC",
        interval="1d",
    )
    
    # Open Interest Stablecoin
    features["open_interest_stablecoin"] = getter.get_history(
        endpoint_name="open_interest_stablecoin",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Open Interest Coin Margin
    features["open_interest_coin_margin"] = getter.get_history(
        endpoint_name="open_interest_coin_margin",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Funding Rate History
    features["funding_rate_history"] = getter.get_history(
        endpoint_name="funding_rate_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Funding Rate OI-Weighted
    features["funding_rate_oi_weight"] = getter.get_history(
        endpoint_name="funding_rate_oi_weight",
        symbol="BTC",
        interval="1d",
    )
    
    # Funding Rate Volume-Weighted
    features["funding_rate_vol_weight"] = getter.get_history(
        endpoint_name="funding_rate_vol_weight",
        symbol="BTC",
        interval="1d",
    )
    
    # Global Long/Short Account Ratio
    features["global_long_short_account_ratio"] = getter.get_history(
        endpoint_name="global_long_short_account_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Top Traders Long/Short Account Ratio
    features["top_long_short_account_ratio"] = getter.get_history(
        endpoint_name="top_long_short_account_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Top Traders Long/Short Position Ratio
    features["top_long_short_position_ratio"] = getter.get_history(
        endpoint_name="top_long_short_position_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Net Position History
    features["net_position"] = getter.get_history(
        endpoint_name="net_position",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Liquidation History
    features["liquidation_history"] = getter.get_history(
        endpoint_name="liquidation_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Liquidation Aggregated
    features["liquidation_aggregated"] = getter.get_history(
        endpoint_name="liquidation_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Orderbook Ask/Bids History
    features["orderbook_ask_bids"] = getter.get_history(
        endpoint_name="orderbook_ask_bids",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Orderbook Aggregated
    features["orderbook_aggregated"] = getter.get_history(
        endpoint_name="orderbook_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # Taker Buy/Sell Volume
    features["taker_buy_sell_volume"] = getter.get_history(
        endpoint_name="taker_buy_sell_volume",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
    )
    
    # Taker Buy/Sell Volume Aggregated
    features["taker_buy_sell_volume_aggregated"] = getter.get_history(
        endpoint_name="taker_buy_sell_volume_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
    )
    
    # =========================================================================
    # Bitcoin On-Chain / Index Features
    # =========================================================================
    
    # Bitcoin Long-Term Holder Supply
    features["bitcoin_lth_supply"] = getter.get_bitcoin_lth_supply(
        pct_window=30,
        z_window=180,
        slope_window=14,
    )
    
    # Bitcoin Active Addresses
    features["bitcoin_active_addresses"] = getter.get_bitcoin_active_addresses(
        pct_window=7,
        z_window=180,
        slope_window=14,
    )
    
    # Bitcoin Short-Term Holder Supply
    features["bitcoin_sth_supply"] = getter.get_bitcoin_sth_supply(
        pct_window=30,
        z_window=180,
        slope_window=14,
    )
    
    # Bitcoin Reserve Risk
    features["bitcoin_reserve_risk"] = getter.get_bitcoin_reserve_risk(
        z_window=180,
        slope_window=14,
    )
    
    # =========================================================================
    # Exchange-Specific Features
    # =========================================================================
    
    # Bitfinex Margin Long/Short
    features["bitfinex_margin_long_short"] = getter.get_bitfinex_margin_long_short(
        symbol="BTC",
        interval="1d",
    )
    
    # Coinbase Premium Index
    features["coinbase_premium_index"] = getter.get_coinbase_premium_index(
        interval="1d",
    )
    
    # =========================================================================
    # Derivatives Indices
    # =========================================================================
    
    # CoinGlass Derivatives Index (CGDI)
    features["cgdi_index"] = getter.get_cgdi_index(
        interval="1d",
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
    
    # Создаём экземпляр FeaturesGetter
    getter = FeaturesGetter(api_key=API_KEY)
    
    # Загружаем все фичи
    print("Загрузка фичей с CoinGlass...")
    features = fetch_all_coinglass_features(getter)
    
    # Выводим сводку
    print_features_summary(features)
    
    # Объединяем в один DataFrame
    print("\nОбъединение фичей...")
    df_merged = merge_all_features(features)
    print(f"Итоговый DataFrame: {len(df_merged)} записей, {len(df_merged.columns)} колонок")
    print(f"Диапазон дат: {df_merged['date'].min()} — {df_merged['date'].max()}")
    print()
    print(df_merged.tail())
