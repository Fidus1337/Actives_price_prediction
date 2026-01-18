import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from dotenv import load_dotenv

## HELPER FUNCTIONS	
from .helpers._coinglass_get_dataframe import _coinglass_get_dataframe, CoinGlassError
from .helpers._coinglass_normalize_time_to_date import _coinglass_normalize_time_to_date
from .helpers._prefix_columns import _prefix_columns

# Load endpoints config from JSON
_ENDPOINTS_PATH = Path(__file__).parent / "features_endpoints.json"
with open(_ENDPOINTS_PATH, "r", encoding="utf-8") as f:
    ENDPOINTS = json.load(f)


class FeaturesGetter:
    """
    Класс для получения исторических данных с CoinGlass API.
    
    Attributes:
        api_key: API ключ CoinGlass
    
    Example:
        >>> getter = FeaturesGetter(api_key="your_api_key")
        >>> df = getter.get_history("open_interest_history")
    """
    
    def __init__(self, api_key: str):
        """
        Инициализирует FeaturesGetter с API ключом.
        
        Args:
            api_key: API ключ CoinGlass
        """
        self.api_key = api_key
    
    def get_history(
        self,
        endpoint_name: str,
        prefix: str | None = None,
        **params,
    ) -> pd.DataFrame:
        """
        Получает исторические данные с CoinGlass API.
        
        Args:
            endpoint_name: Имя эндпоинта из ENDPOINTS (например, "open_interest_history")
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
            api_key=self.api_key,
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
    
    @staticmethod
    def list_endpoints() -> list[str]:
        """Возвращает список доступных эндпоинтов."""
        return sorted(ENDPOINTS.keys())
    
    def get_bitcoin_lth_supply(
        self,
        pct_window: int = 30,
        z_window: int = 180,
        slope_window: int = 14,
        prefix: str = "index_btc_lth_supply",
    ) -> pd.DataFrame:
        """
        Bitcoin Long-Term Holder Supply с расчётными фичами для прогнозирования.
        
        Args:
            pct_window: Окно для процентного изменения (дней)
            z_window: Окно для z-score (дней)
            slope_window: Окно для slope/velocity (дней)
            prefix: Префикс для колонок
        
        Returns:
            DataFrame с колонками:
              - date
              - {prefix}__price
              - {prefix}__lth_supply
              - {prefix}__supply_pct{pct_window}
              - {prefix}__supply_z{z_window}
              - {prefix}__supply_slope{slope_window}
        """
        df = _coinglass_get_dataframe(
            endpoint="/index/bitcoin-long-term-holder-supply",
            api_key=self.api_key,
        )
        
        if df.empty:
            return df
        
        # timestamp -> date (этот эндпоинт использует timestamp, а не time)
        if "timestamp" in df.columns:
            df["date"] = _coinglass_normalize_time_to_date(df["timestamp"])
            df = df.drop(columns=["timestamp"])
        
        # Нормализация числовых колонок
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["lth_supply"] = pd.to_numeric(df.get("long_term_holder_supply"), errors="coerce")
        
        # Очистка и сортировка
        df = (
            df[["date", "price", "lth_supply"]]
            .dropna(subset=["date"])
            .sort_values("date", kind="stable")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        
        s = df["lth_supply"].astype(float)
        
        # Feature 1: pct change over N days (supply expansion/contraction proxy)
        df[f"supply_pct{pct_window}"] = s / s.shift(pct_window) - 1.0
        
        # Feature 2: rolling z-score (regime-normalized supply)
        minp = max(30, z_window // 3)
        roll = s.rolling(z_window, min_periods=minp)
        mu = roll.mean()
        sd = roll.std(ddof=0).replace(0.0, np.nan)
        df[f"supply_z{z_window}"] = (s - mu) / sd
        
        # Feature 3: slope / velocity
        df[f"supply_slope{slope_window}"] = s.diff(slope_window) / float(slope_window)
        
        # Префикс
        df = _prefix_columns(df, prefix=prefix, keep=("date",))
        
        return df


# ============================================================================
# Примеры использования
# ============================================================================
if __name__ == "__main__":
    load_dotenv("dev.env")
    API_KEY = os.getenv("COINGLASS_API_KEY")
    
    if not API_KEY:
        raise ValueError("COINGLASS_API_KEY не найден в dev.env")
    
    # Создаём экземпляр FeaturesGetter
    getter = FeaturesGetter(api_key=API_KEY)
    
    print("Доступные эндпоинты:")
    for name in getter.list_endpoints():
        print(f"  - {name}")
    print()
    
    # -------------------------------------------------------------------------
    # Пример 1: Open Interest History
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 1: Open Interest History (с дефолтными параметрами)")
    print("=" * 60)
    
    try:
        df = getter.get_history("open_interest_history")
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
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
        df = getter.get_history(
            "funding_rate_history",
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
        df = getter.get_history(
            "global_long_short_account_ratio",
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
        df = getter.get_history(
            "open_interest_aggregated",
            symbol="ETH",
        )
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.head())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    # -------------------------------------------------------------------------
    # Пример 5: Bitcoin LTH Supply
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 5: Bitcoin Long-Term Holder Supply")
    print("=" * 60)
    
    try:
        df = getter.get_bitcoin_lth_supply()
        print(f"Получено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(df.tail())
        print()
    except CoinGlassError as e:
        print(f"Ошибка: {e}\n")
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)
