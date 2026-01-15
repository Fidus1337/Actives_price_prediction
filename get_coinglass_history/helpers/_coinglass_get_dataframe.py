import requests
import pandas as pd
import os
from dotenv import load_dotenv

BASE_URL = "https://open-api-v4.coinglass.com/api"

class CoinGlassError(RuntimeError):
    pass


def _coinglass_get_dataframe(
    endpoint: str,
    api_key: str,
    params: dict | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Универсальный клиент: дергает endpoint, проверяет code/msg, возвращает DataFrame(data).
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {"accept": "application/json", "CG-API-KEY": api_key}

    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise CoinGlassError(f"HTTP error {r.status_code}: {r.text[:300]}") from e

    j = r.json()

    # CoinGlass обычно возвращает code как строку: "0" = success
    code = str(j.get("code", ""))
    if code != "0":
        raise CoinGlassError(f"CoinGlass error code={code}, msg={j.get('msg')}")

    data = j.get("data")
    if data is None:
        # иногда data может быть [] — это ок; None — подозрительно
        raise CoinGlassError("Response has no 'data' field")

    df = pd.DataFrame(data)
    return df


# ============================================================================
# Примеры использования
# ============================================================================
if __name__ == "__main__":
    # Загружаем API ключ из dev.env
    load_dotenv("../../dev.env")
    API_KEY = os.getenv("COINGLASS_API_KEY")
    
    if not API_KEY:
        raise ValueError("COINGLASS_API_KEY не найден в dev.env")
    
    # -------------------------------------------------------------------------
    # Пример 1: Получение списка поддерживаемых монет
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 1: Список поддерживаемых монет (Supported Coins)")
    print("=" * 60)
    
    try:
        df_coins = _coinglass_get_dataframe(
            endpoint="/futures/supported-coins",
            api_key=API_KEY,
        )
        print(f"Найдено {len(df_coins)} монет")
        print(df_coins.head(10))
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}")
    
    # -------------------------------------------------------------------------
    # Пример 2: Open Interest по Bitcoin
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 2: Open Interest для BTC")
    print("=" * 60)
    
    try:
        df_oi = _coinglass_get_dataframe(
            endpoint="/futures/open-interest/history",
            api_key=API_KEY,
            params={
                "exchange": "Binance",
                "symbol": "BTCUSDT",
                "interval": "1d",
            },
        )
        print(f"Получено {len(df_oi)} записей")
        print(df_oi.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}")
    
    # -------------------------------------------------------------------------
    # Пример 3: Funding Rate по Ethereum
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 3: Funding Rate для ETH")
    print("=" * 60)
    
    try:
        df_funding = _coinglass_get_dataframe(
            endpoint="/futures/funding-rate/history",
            api_key=API_KEY,
            params={
                "exchange": "Binance",
                "symbol": "ETHUSDT",
                "interval": "1d",
            },
        )
        print(f"Получено {len(df_funding)} записей")
        print(df_funding.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}")
    
    # -------------------------------------------------------------------------
    # Пример 4: Liquidation данные
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 4: Liquidation данные для BTC")
    print("=" * 60)
    
    try:
        df_liq = _coinglass_get_dataframe(
            endpoint="/futures/liquidation/history",
            api_key=API_KEY,
            params={
                "exchange": "Binance",
                "symbol": "BTCUSDT",
                "interval": "1d",
            },
        )
        print(f"Получено {len(df_liq)} записей")
        print(df_liq.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}")
    
    # -------------------------------------------------------------------------
    # Пример 5: Long/Short Ratio
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 5: Long/Short Ratio для BTC")
    print("=" * 60)
    
    try:
        df_ls = _coinglass_get_dataframe(
            endpoint="/futures/global-long-short-account-ratio/history",
            api_key=API_KEY,
            params={
                "exchange": "Binance",
                "symbol": "BTCUSDT",
                "interval": "1d",
            },
        )
        print(f"Получено {len(df_ls)} записей")
        print(df_ls.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}")
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)
