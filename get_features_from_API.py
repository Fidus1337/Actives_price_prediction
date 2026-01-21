from FeaturesGetterModule.FeaturesGetter import FeaturesGetter

def get_features(getter: FeaturesGetter, API_KEY: str):

    # Open Interest History
    df_oi = getter.get_history(
            endpoint_name="open_interest_history",
            exchange="Binance",
            symbol="BTCUSDT",
            interval="1d",
            prefix="futures_open_interest_history"
        )
    
    # Open Interest Aggregated
    df_oi_agg = getter.get_history(
    endpoint_name="open_interest_aggregated",
    symbol="BTC",
    interval="1d",
    prefix="futures_open_interest_aggregated_history"
    )
    
        # Open Interest Stablecoin
    df_stable_oi = getter.get_history(
        endpoint_name="open_interest_stablecoin",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
        prefix="futures_open_interest_aggregated_stablecoin_history"
    )

    # Open Interest Stablecoin
    df_coin_margin = getter.get_history(
        endpoint_name="open_interest_coin_margin",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
        prefix="futures_open_interest_aggregated_coin_margin_history"
    )

    # Funding Rate History
    df_funding = getter.get_history(
        endpoint_name="funding_rate_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_funding_rate_history"
    )

    # Funding Rate OI-Weighted
    df_oi_weight_funding = getter.get_history(
        endpoint_name="funding_rate_oi_weight",
        symbol="BTC",
        interval="1d",
        prefix="futures_funding_rate_oi_weight_history"
    )

    # Funding Rate Volume-Weighted
    df_vol_weight_funding = getter.get_history(
            endpoint_name="funding_rate_vol_weight",
            symbol="BTC",
            interval="1d",
            prefix="futures_funding_rate_vol_weight_history"
        )

    # Global Long/Short Account Ratio
    df_ls_accounts = getter.get_history(
        endpoint_name="global_long_short_account_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_global_long_short_account_ratio_history"
    )

    # Top Traders Long/Short Account Ratio
    df_top_ls_accounts = getter.get_history(
        endpoint_name="top_long_short_account_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_top_long_short_account_ratio_history"
    )

    # Top Traders Long/Short Position Ratio
    df_top_ls_positions = getter.get_history(
        endpoint_name="top_long_short_position_ratio",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_top_long_short_position_ratio_history"
    )

    # Net Position History
    df_net_pos = getter.get_history(
        endpoint_name="net_position",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_v2_net_position_history"
    )

    # Liquidation History
    df_liq = getter.get_history(
        endpoint_name="liquidation_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_liquidation_history"
    )

    # Liquidation Aggregated
    df_liq_agg = getter.get_history(
        endpoint_name="liquidation_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
        prefix="futures_liquidation_aggregated_history"
    )

    # Orderbook Ask/Bids History
    df_ob = getter.get_history(
        endpoint_name="orderbook_ask_bids",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_orderbook_ask_bids_history"
    )

    # Orderbook Aggregated
    df_ob_agg = getter.get_history(
        endpoint_name="orderbook_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
        prefix="futures_orderbook_aggregated_ask_bids_history"
    )

    # Taker Buy/Sell Volume
    df_taker = getter.get_history(
        endpoint_name="taker_buy_sell_volume",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix="futures_v2_taker_buy_sell_volume_history"
    )

    # Bitcoin Long-Term Holder Supply
    df_lth_supply = getter.get_bitcoin_lth_supply(
        pct_window=30,
        z_window=180,
        slope_window=14,
        prefix="index_btc_lth_supply"
    )

    # Bitcoin Active Addresses
    df_aa = getter.get_bitcoin_active_addresses(
        pct_window=7,
        z_window=180,
        slope_window=14,
        prefix="index_btc_active_addresses"
    )

    # Bitcoin Short-Term Holder Supply
    df_sth_supply = getter.get_bitcoin_sth_supply(
        pct_window=30,
        z_window=180,
        slope_window=14,
        prefix="index_btc_sth_supply"
    )

    # Taker Buy/Sell Volume Aggregated
    df_taker_agg = getter.get_history(
        endpoint_name="taker_buy_sell_volume_aggregated",
        exchange_list="Binance",
        symbol="BTC",
        interval="1d",
        prefix="futures_aggregated_taker_buy_sell_volume_history"
    )

    # Bitfinex Margin Long/Short
    bitfinex_margin_ls_df = getter.get_bitfinex_margin_long_short(
        symbol="BTC",
        interval="1d",
        prefix="bfx_margin"
    )

    new_names = {
        "bfx_margin__long_quantity": "long_quantity",
        "bfx_margin__short_quantity": "short_quantity"
    }

    bitfinex_margin_ls_df = bitfinex_margin_ls_df.rename(columns=new_names)

    # CoinGlass Derivatives Index (CGDI)
    futures_cgdi_index_df = getter.get_cgdi_index(
        interval="1d",
        prefix="cgdi"
    )
    new_names = {
        "cgdi__index_value": "cgdi_index_value",
        "cgdi__log_level": "cgdi_log_level",
        "cgdi__dev_from_base": "cgdi_dev_from_base",
        "cgdi__dev_softsign": "cgdi_dev_softsign",
    }
    futures_cgdi_index_df = futures_cgdi_index_df.rename(columns=new_names)

    # Coinbase Premium Index
    coinbase_premium_df = getter.get_coinbase_premium_index(
        interval="1d",
        prefix="premium"
    )
    new_names = {
        "premium__premium": "premium",
        "premium__premium_rate": "premium_rate",
        "premium__premium_abs": "cb_premium_abs",
        "premium__premium_softsign": "cb_premium_softsign",
        "premium__premium_rate_bps": "cb_premium_rate_bps",
        "premium__implied_ref_price": "cb_implied_ref_price",
    }
    coinbase_premium_df = coinbase_premium_df.rename(columns=new_names)

    # Bitcoin Reserve Risk
    df_rr = getter.get_bitcoin_reserve_risk(
        z_window=180,
        slope_window=14,
        prefix="index_btc_reserve_risk"
    )

    # Spot Price History (OHLCV)
    df_spot = getter.get_history(
        endpoint_name="spot_price_history",
        exchange="Binance",
        symbol="BTCUSDT",
        interval="1d",
        prefix=""
    )
    df_spot.columns = df_spot.columns.str.lstrip("_")

    return [
    df_oi,
    df_oi_agg,
    df_stable_oi,
    df_coin_margin,
    df_funding,
    df_oi_weight_funding,
    df_vol_weight_funding,
    df_ls_accounts,
    df_top_ls_accounts,
    df_top_ls_positions,
    df_net_pos,
    df_liq,
    df_liq_agg,
    df_ob,
    df_ob_agg,
    df_taker,
    df_taker_agg,
    bitfinex_margin_ls_df,
    futures_cgdi_index_df,
    coinbase_premium_df,
    df_lth_supply,
    df_aa,
    df_sth_supply,
    df_rr,
    df_spot
    ]


