"""
Скрипт для просмотра эндпоинтов CoinGlass: Новости и Экономический календарь.
Независимый файл — просто дёргает API и выводит результат с фильтрацией по дате.

Использование:
    python explore_coinglass_news.py
    python explore_coinglass_news.py --date 2026-03-01 --window 7
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

BASE_URL = "https://open-api-v4.coinglass.com/api"
API_KEY = os.getenv("COINGLASS_API_KEY")


def coinglass_get_raw(endpoint: str, params: dict | None = None) -> dict:
    """Дёргает CoinGlass endpoint и возвращает сырой JSON-ответ."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}

    r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def _ts_ms_to_date(ts_ms: int) -> datetime:
    """Unix timestamp в миллисекундах → datetime UTC."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _parse_release_time(value) -> datetime | None:
    """Пробует распарсить article_release_time (может быть unix ms или строка)."""
    if value is None:
        return None
    # Unix timestamp в миллисекундах (число или строка-число)
    if isinstance(value, (int, float)):
        return _ts_ms_to_date(int(value))
    if isinstance(value, str) and value.isdigit():
        return _ts_ms_to_date(int(value))
    # ISO-строка или другой формат
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _strip_html(html: str) -> str:
    """Убирает HTML-теги, оставляет чистый текст."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_articles_paginated(
    dt_from: datetime,
    dt_to: datetime,
    max_pages: int = 50,
) -> list[dict]:
    """Загружает статьи с пагинацией, возвращает попавшие в [dt_from, dt_to]."""
    results: list[dict] = []
    pages_fetched = 0

    for page in range(1, max_pages + 1):
        resp = coinglass_get_raw("/article/list", {"page": page})
        data = resp.get("data")
        if not data or not isinstance(data, list):
            break

        pages_fetched = page
        oldest_on_page: datetime | None = None

        for item in data:
            dt = _parse_release_time(item.get("article_release_time"))
            if dt is None:
                continue
            if oldest_on_page is None or dt < oldest_on_page:
                oldest_on_page = dt
            if dt_from <= dt <= dt_to:
                results.append(item)

        if oldest_on_page is not None and oldest_on_page < dt_from:
            break

    print(f"  Загружено {pages_fetched} стр., {len(results)} статей в окне")

    if pages_fetched >= max_pages and not results:
        print(f"  WARN: окно за пределами истории API (~23 дня)")

    return results


# =========================================================================
#  НОВОСТИ
# =========================================================================
def explore_news(date_from: str, date_to: str):
    """Достаёт новости и фильтрует по дате."""
    dt_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print("=" * 70)
    print(f"  НОВОСТИ (article/list)  |  {date_from} → {date_to}")
    print("=" * 70)

    filtered = _fetch_articles_paginated(dt_from, dt_to)

    # Добавляем _parsed_date для отображения
    for item in filtered:
        item["_parsed_date"] = _parse_release_time(item.get("article_release_time"))

    if not filtered:
        print(f"  Ни одна новость не попала в окно {date_from} → {date_to}")
        return

    print(f"  В окне {date_from} → {date_to}: {len(filtered)} статей\n")

    # Сортируем по дате (новые сверху)
    filtered.sort(key=lambda x: x.get("_parsed_date") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    for i, item in enumerate(filtered):
        dt = item.get("_parsed_date")
        date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "???"
        title = item.get("article_title", "—")
        source = item.get("source_name", "—")
        content = _strip_html(item.get("article_content", ""))[:300]

        print(f"  [{i + 1}] {date_str}  |  {source}")
        print(f"      {title}")
        print(f"      {content}...")
        print()


# =========================================================================
#  ЭКОНОМИЧЕСКИЙ КАЛЕНДАРЬ
# =========================================================================
IMPORTANCE_LABELS = {1: "Minor", 2: "Medium", 3: "Major"}


def explore_economic_calendar(date_from: str, date_to: str):
    """Достаёт экономический календарь и фильтрует по дате + важности."""
    dt_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print("\n" + "=" * 70)
    print(f"  ЭКОНОМИЧЕСКИЙ КАЛЕНДАРЬ  |  {date_from} → {date_to}")
    print("=" * 70)

    resp = coinglass_get_raw("/calendar/economic-data")
    data = resp.get("data", [])
    print(f"\n  Всего получено: {len(data)} событий")

    # Фильтруем по дате
    in_window = []
    for item in data:
        ts = item.get("publish_timestamp")
        if ts is None:
            continue
        dt = _ts_ms_to_date(ts)
        if dt_from <= dt <= dt_to:
            item["_parsed_date"] = dt
            in_window.append(item)

    if not in_window:
        # Показать диапазон дат в данных
        all_dates = [_ts_ms_to_date(it["publish_timestamp"]) for it in data if it.get("publish_timestamp")]
        if all_dates:
            print(f"  Диапазон дат в ответе: {min(all_dates).strftime('%Y-%m-%d')} → {max(all_dates).strftime('%Y-%m-%d')}")
        print(f"  Ни одно событие не попало в окно {date_from} → {date_to}")
        return

    # Сортируем по дате
    in_window.sort(key=lambda x: x["_parsed_date"])

    # Группируем: сначала Major/Medium, потом Minor
    major_medium = [it for it in in_window if it.get("importance_level", 1) >= 2]
    minor = [it for it in in_window if it.get("importance_level", 1) < 2]

    print(f"  В окне: {len(in_window)} событий (Major/Medium: {len(major_medium)}, Minor: {len(minor)})")

    # Выводим Major/Medium
    if major_medium:
        print(f"\n  --- ВАЖНЫЕ СОБЫТИЯ (importance ≥ 2) ---\n")
        for item in major_medium:
            _print_calendar_item(item)

    # Выводим Minor (сокращённо)
    if minor:
        print(f"\n  --- MINOR СОБЫТИЯ ({len(minor)} шт, первые 10) ---\n")
        for item in minor[:10]:
            _print_calendar_item(item, short=True)


def _print_calendar_item(item: dict, short: bool = False):
    dt = item.get("_parsed_date")
    date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "???"
    name = item.get("calendar_name", "—")
    country = item.get("country_name", "—")
    importance = IMPORTANCE_LABELS.get(item.get("importance_level", 1), "?")
    effect = item.get("data_effect", "—")

    forecast = item.get("forecast_value", "")
    previous = item.get("previous_value", "")
    actual = item.get("published_value", "")

    if short:
        print(f"    {date_str}  [{country}] {name}")
    else:
        print(f"  {date_str}  [{importance}] [{country}] {name}")
        print(f"    Effect: {effect}  |  Forecast: {forecast or '—'}  |  Previous: {previous or '—'}  |  Actual: {actual or '—'}")
        print()


# =========================================================================
#  MAIN
# =========================================================================
def main():
    if not API_KEY:
        print("COINGLASS_API_KEY не найден в dev.env")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Просмотр CoinGlass News & Economic Calendar")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Дата начала (YYYY-MM-DD), по умолчанию сегодня")
    parser.add_argument("--window", type=int, default=7,
                        help="Окно в днях (по умолчанию 7)")
    args = parser.parse_args()

    date_from = args.date
    date_to = (datetime.strptime(date_from, "%Y-%m-%d") + timedelta(days=args.window)).strftime("%Y-%m-%d")

    print(f"CoinGlass API Explorer | Дата: {date_from}, окно: {args.window}d → до {date_to}")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}\n")

    explore_news(date_from, date_to)
    explore_economic_calendar(date_from, date_to)

    print("\n\nГотово!")


if __name__ == "__main__":
    main()
