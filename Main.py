from dotenv import load_dotenv
import os

from Getting_and_preprocessing_features import (
    fetch_all_coinglass_features,
    merge_all_features,
    print_features_summary,
)

## ENV VARIABLES

load_dotenv("dev.env")

BASE_URL = os.getenv("COIN_GLASS_ENDPOINT")
API_KEY = os.getenv("COINGLASS_API_KEY")

if not API_KEY:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")


if __name__ == "__main__":
    # Загружаем все фичи с CoinGlass
    print("Загрузка фичей с CoinGlass API...")
    features = fetch_all_coinglass_features(API_KEY)
    
    # Выводим сводку
    print_features_summary(features)
    
    # Доступ к отдельным DataFrame
    df_oi = features["open_interest_history"]
    df_oi_agg = features["open_interest_aggregated"]
    df_stable_oi = features["open_interest_stablecoin"]
    df_coin_margin = features["open_interest_coin_margin"]
    df_funding = features["funding_rate_history"]
    df_oi_weight_funding = features["funding_rate_oi_weight"]
    df_vol_weight_funding = features["funding_rate_vol_weight"]
    df_ls_accounts = features["global_long_short_account_ratio"]
    df_top_ls_accounts = features["top_long_short_account_ratio"]
    df_net_pos = features["net_position"]
    df_liq = features["liquidation_history"]
    df_liq_agg = features["liquidation_aggregated"]
    df_ob = features["orderbook_ask_bids"]
    df_ob_agg = features["orderbook_aggregated"]
    df_taker = features["taker_buy_sell_volume"]
    df_lth_supply = features["bitcoin_lth_supply"]
    
    # Объединяем все фичи в один DataFrame
    print("\nОбъединение всех фичей...")
    df_all = merge_all_features(features)
    
    print(f"\nИтоговый DataFrame:")
    print(f"  Записей: {len(df_all)}")
    print(f"  Колонок: {len(df_all.columns)}")
    print(f"  Диапазон дат: {df_all['date'].min()} — {df_all['date'].max()}")
    print()
    print(df_all.tail())
