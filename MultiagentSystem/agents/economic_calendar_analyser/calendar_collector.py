"""
Incremental CoinGlass economic calendar collector.

Fetches macro-economic events from CoinGlass API (~28 day rolling window)
and merges them into a local JSON archive, deduplicating by
(calendar_name, publish_timestamp).

No classification at this stage — just raw data archiving.
Classification and LLM aggregation will be added in later steps.

Usage:
    python -m MultiagentSystem.agents.economic_calendar_analyser.calendar_collector
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / "dev.env")

BASE_URL = "https://open-api-v4.coinglass.com/api"
API_KEY = os.getenv("COINGLASS_API_KEY")
ARCHIVE_PATH = Path(__file__).parent / "calendar_archive.json"


def _make_key(event: dict) -> str:
    """Unique key for deduplication: calendar_name + publish_timestamp."""
    name = event.get("calendar_name", "")
    ts = event.get("publish_timestamp", "")
    return f"{name}||{ts}"


def _ts_to_date_str(ts_ms) -> str | None:
    """Unix timestamp (ms) -> 'YYYY-MM-DD' string."""
    if ts_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def _ts_to_datetime(ts_ms) -> datetime | None:
    """Unix timestamp (ms) -> datetime UTC."""
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _prepare_for_archive(event: dict) -> dict:
    """Add human-readable date field before storing."""
    cleaned = dict(event)
    date_str = _ts_to_date_str(cleaned.get("publish_timestamp"))
    if date_str:
        cleaned["date"] = date_str
    return cleaned


def _load_archive() -> list[dict]:
    """Load existing archive or return empty list. Backfills missing 'date' field."""
    if ARCHIVE_PATH.exists():
        articles = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
        for a in articles:
            if "date" not in a:
                date_str = _ts_to_date_str(a.get("publish_timestamp"))
                if date_str:
                    a["date"] = date_str
        return articles
    return []


def _save_archive(events: list[dict]) -> None:
    """Save archive sorted by date (newest first)."""
    def sort_key(e):
        dt = _ts_to_datetime(e.get("publish_timestamp"))
        return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

    events.sort(key=sort_key, reverse=True)

    ARCHIVE_PATH.write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fetch_from_api() -> list[dict]:
    """Fetch all economic calendar events from CoinGlass (single call, no pagination)."""
    if not API_KEY:
        print("[calendar_collector] ERROR: COINGLASS_API_KEY not found in dev.env")
        return []

    url = f"{BASE_URL}/calendar/economic-data"
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])

    print(f"[calendar_collector] Fetched {len(data)} events from API")
    return data


def collect_calendar_events() -> dict:
    """
    Main collection function.
    Fetches all available events from API, deduplicates against
    existing archive, and saves.

    Returns stats: {before, fetched, new, after, date_range}.
    """
    archive = _load_archive()
    existing_keys = {_make_key(e) for e in archive}
    before_count = len(archive)

    fresh = _fetch_from_api()

    new_count = 0
    for event in fresh:
        key = _make_key(event)
        if key not in existing_keys:
            archive.append(_prepare_for_archive(event))
            existing_keys.add(key)
            new_count += 1

    _save_archive(archive)

    dates = [_ts_to_datetime(e.get("publish_timestamp")) for e in archive]
    dates = [d for d in dates if d is not None]

    date_range = (
        f"{min(dates).strftime('%Y-%m-%d')} -> {max(dates).strftime('%Y-%m-%d')}"
        if dates else "empty"
    )

    stats = {
        "before": before_count,
        "fetched": len(fresh),
        "new": new_count,
        "after": len(archive),
        "date_range": date_range,
    }

    print(f"[calendar_collector] Archive: {before_count} -> {len(archive)} events (+{new_count} new)")
    print(f"[calendar_collector] Date range: {date_range}")

    return stats


def get_events_in_range(
    dt_from: datetime,
    dt_to: datetime,
) -> list[dict]:
    """
    Read events from the local archive within [dt_from, dt_to].
    """
    archive = _load_archive()
    results = []
    for event in archive:
        dt = _ts_to_datetime(event.get("publish_timestamp"))
        if dt and dt_from <= dt <= dt_to:
            results.append(event)

    results.sort(
        key=lambda e: _ts_to_datetime(e.get("publish_timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return results


if __name__ == "__main__":
    stats = collect_calendar_events()
    print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")
