import pandas as pd
import os
import json
from pathlib import Path
from dotenv import load_dotenv

## HELPER FUNCTIONS	
from helpers._coinglass_get_dataframe import _coinglass_get_dataframe, CoinGlassError
from helpers._coinglass_normalize_time_to_date import _coinglass_normalize_time_to_date
from helpers._prefix_columns import _prefix_columns

# Load endpoints config from JSON
_ENDPOINTS_PATH = Path(__file__).parent / "features_endpoints.json"
with open(_ENDPOINTS_PATH, "r", encoding="utf-8") as f:
    ENDPOINTS = json.load(f)


def get_coinglass_history(
    endpoint_name: str,
    api_key: str,
    prefix: str | None = None,
    **params,
) -> pd.DataFrame:
    """
    Универсальная функция для получения исторических данных с CoinGlass API.
    
    Args:
        endpoint_name: Имя эндпоинта из ENDPOINTS (например, "open_interest_history")
        api_key: API ключ CoinGlass
        prefix: Префикс для колонок (по умолчанию = endpoint_name)
        **params: Параметры запроса (exchange, symbol, interval и т.д.)
                  Если не указаны, используются default_params из конфига.
    
    Returns:
        DataFrame с колонкой date и данными с префиксами.
    
    Raises:
        ValueError: Если endpoint_name не найден в ENDPOINTS
        CoinGlassError: При ошибках API
    """
    if endpoint_name not in ENDPOINTS:
        available = ", ".join(sorted(ENDPOINTS.keys()))
        raise ValueError(f"Unknown endpoint: '{endpoint_name}'. Available: {available}")
    
    cfg = ENDPOINTS[endpoint_name]
    
    # Merge default params with user params (user params override defaults)
    request_params = {**cfg["default_params"], **params}
    
    # Fetch data
    df = _coinglass_get_dataframe(
        endpoint=cfg["path"],
        api_key=api_key,
        params=request_params,
    )
    
    if df.empty:
        return df
    
    # time -> date
    if "time" in df.columns:
        df["date"] = _coinglass_normalize_time_to_date(df["time"])
        df = df.drop(columns=["time"])
    else:
        df["date"] = pd.NA
    
    # Convert all non-date columns to numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add prefix
    if prefix is None:
        prefix = endpoint_name
    
    df = _prefix_columns(df, prefix=prefix, keep=("date",))
    
    return df


def list_endpoints() -> list[str]:
    """Возвращает список доступных эндпоинтов."""
    return sorted(ENDPOINTS.keys())


# ============================================================================
# Примеры использования
# ============================================================================
if __name__ == "__main__":
    load_dotenv("../dev.env")
    API_KEY = os.getenv("COINGLASS_API_KEY")
    
    if not API_KEY:
        raise ValueError("COINGLASS_API_KEY не найден в dev.env")
    
    print("Доступные эндпоинты:")
    for name in list_endpoints():
        print(f"  - {name}")
    print()
    
    # -------------------------------------------------------------------------
    # Пример 1: Open Interest History
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 1: Open Interest History (с дефолтными параметрами)")
    print("=" * 60)
    
    try:
        df = get_coinglass_history("open_interest_history", api_key=API_KEY)
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 2: Funding Rate с кастомными параметрами
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 2: Funding Rate для ETH")
    print("=" * 60)
    
    try:
        df = get_coinglass_history(
            "funding_rate_history",
            api_key=API_KEY,
            symbol="ETHUSDT",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 3: Long/Short Ratio с кастомным префиксом
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 3: Long/Short Ratio с кастомным префиксом")
    print("=" * 60)
    
    try:
        df = get_coinglass_history(
            "global_long_short_account_ratio",
            api_key=API_KEY,
            prefix="ls_ratio",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 4: Aggregated данные (без exchange)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 4: Open Interest Aggregated")
    print("=" * 60)
    
    try:
        df = get_coinglass_history(
            "open_interest_aggregated",
            api_key=API_KEY,
            symbol="ETH",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)
