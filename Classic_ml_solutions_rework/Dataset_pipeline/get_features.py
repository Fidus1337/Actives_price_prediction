"""
FeaturesGetterRework — самодостаточный клиент для получения фич с CoinGlass v4.

Конфиг эндпоинтов лежит рядом, в `features_endpoints.json`.
Каждая запись описывает один логический "feature":
    {
        "<feature_name>": {
            "path":           "/spot/price/history",          # путь после BASE_URL
            "default_params": {"exchange": "Bybit", ...},     # параметры запроса
            "prefix":         "btc_spot"                       # префикс колонок (опц.)
        },
        ...
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = "https://open-api-v4.coinglass.com/api"

_HERE = Path(__file__).resolve().parent
_DEFAULT_ENDPOINTS_PATH = _HERE / "features_endpoints.json"


class CoinGlassError(RuntimeError):
    """Ошибка ответа CoinGlass (HTTP / некорректный code в JSON)."""


class FeaturesGetterRework:
    """
    Простой ретривер фич с CoinGlass.

    Example:
        >>> getter = FeaturesGetterRework()
        >>> df = getter.get_feature("spot_price_history")
        >>> df = getter.get_feature("spot_price_history", symbol="ETHUSDT", limit=500)
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoints_path: str | Path | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        if api_key is None:
            if env_path is not None:
                load_dotenv(env_path)
            else:
                load_dotenv()
            api_key = os.getenv("COINGLASS_API_KEY")
        if not api_key:
            raise ValueError(
                "COINGLASS_API_KEY не задан: передай api_key или положи его в .env"
            )
        self.api_key = api_key

        self.endpoints_path = Path(endpoints_path or _DEFAULT_ENDPOINTS_PATH).resolve()
        self.endpoints: dict[str, dict[str, Any]] = self._load_endpoints(self.endpoints_path)

    @staticmethod
    def _load_endpoints(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Не найден конфиг эндпоинтов: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} должен быть JSON-объектом верхнего уровня")
        return data

    def list_features(self) -> list[str]:
        return sorted(self.endpoints.keys())

    def get_feature(
        self,
        name: str,
        prefix: str | None = None,
        **param_overrides: Any,
    ) -> pd.DataFrame:
        """
        Тянет фичу по её имени из `features_endpoints.json`.

        Args:
            name:            ключ из конфига (например, "spot_price_history")
            prefix:          переопределяет prefix из конфига (None = брать из json
                             или, если там нет — само имя фичи)
            **param_overrides: любые поля, которые перезапишут default_params
                               (exchange, symbol, interval, limit, start_time, ...)

        Returns:
            DataFrame с колонкой `date` и числовыми колонками с префиксом.
            Пустой DataFrame, если CoinGlass вернул пустой data.
        """
        if name not in self.endpoints:
            available = ", ".join(self.list_features()) or "(пусто)"
            raise ValueError(f"Неизвестная фича '{name}'. Доступно: {available}")

        cfg = self.endpoints[name]
        path = cfg["path"]
        request_params = {**cfg.get("default_params", {}), **param_overrides}
        effective_prefix = prefix or cfg.get("prefix") or name

        raw = self._coinglass_get(path, params=request_params)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        if df.empty:
            return df

        df = self._normalize_time_to_date(df)

        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["date"])
              .sort_values("date", kind="stable")
              .drop_duplicates(subset=["date"], keep="last")
              .reset_index(drop=True)
        )

        df = self._prefix_columns(df, prefix=effective_prefix, keep=("date",))
        return df

    def _coinglass_get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: int = 20,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> list[dict[str, Any]]:
        url = f"{BASE_URL}{endpoint}"
        headers = {"accept": "application/json", "CG-API-KEY": self.api_key}

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
                try:
                    r.raise_for_status()
                except requests.HTTPError as e:
                    raise CoinGlassError(f"HTTP {r.status_code}: {r.text[:300]}") from e

                j = r.json()
                code = str(j.get("code", ""))
                if code != "0":
                    msg = j.get("msg")
                    exc = CoinGlassError(f"CoinGlass code={code}, msg={msg}")
                    if code == "500" and attempt < retries - 1:
                        last_exc = exc
                        wait = retry_delay * (2 ** attempt)
                        print(f"  [retry {attempt + 1}/{retries}] {endpoint} -> code=500, sleep {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    raise exc

                data = j.get("data")
                if data is None:
                    raise CoinGlassError("Ответ без поля 'data'")
                return data

            except CoinGlassError:
                raise
            except requests.RequestException as e:
                last_exc = CoinGlassError(f"Network error: {e}")
                if attempt < retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  [retry {attempt + 1}/{retries}] {endpoint} -> {e}, sleep {wait:.0f}s")
                    time.sleep(wait)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _normalize_time_to_date(df: pd.DataFrame) -> pd.DataFrame:
        """Приводит поле time/timestamp -> колонка `date` (datetime64[ns], без tz)."""
        time_col = next((c for c in ("time", "timestamp") if c in df.columns), None)
        if time_col is None:
            df["date"] = pd.NaT
            return df

        series = pd.to_numeric(df[time_col], errors="coerce")
        unit = "ms" if series.dropna().median() and series.dropna().median() > 1e11 else "s"
        df["date"] = pd.to_datetime(series, unit=unit, utc=True).dt.tz_convert(None).dt.normalize()
        return df.drop(columns=[time_col])

    @staticmethod
    def _prefix_columns(
        df: pd.DataFrame,
        prefix: str,
        keep: tuple[str, ...] = ("date",),
    ) -> pd.DataFrame:
        rename_map = {c: f"{prefix}__{c}" for c in df.columns if c not in keep}
        return df.rename(columns=rename_map)


if __name__ == "__main__":
    getter = FeaturesGetterRework(env_path=_HERE.parent.parent / "dev.env")

    print("Доступные фичи:")
    for name in getter.list_features():
        print(f"  - {name}")
    print()

    df = getter.get_feature("spot_price_history")
    print(f"shape: {df.shape}")
    print(f"cols : {list(df.columns)}")
    print(df.tail())
