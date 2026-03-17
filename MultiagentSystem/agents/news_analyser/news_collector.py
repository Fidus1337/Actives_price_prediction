"""
Incremental CoinGlass news collector.

Fetches the latest articles from CoinGlass API (~24 day rolling window)
and merges them into a local JSON archive, deduplicating by
(article_title, article_release_time).

Usage:
    cd MultiagentSystem
    python agents/news_analyser/news_collector.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Enable standalone execution (same pattern as multiagent_system_main.py)
_MULTIAGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MULTIAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIAGENT_ROOT))

from agents.news_analyser.helpers import (
    coinglass_get_raw,
    fetch_articles_in_window,
    parse_release_time,
    strip_html,
)

ARCHIVE_PATH = Path(__file__).parent / "news_archive.json"


def _make_key(article: dict) -> str:
    """Unique key for deduplication: title + release timestamp."""
    title = article.get("article_title", "")
    ts = article.get("article_release_time", "")
    return f"{title}||{ts}"


def _prepare_for_archive(article: dict) -> dict:
    """Strip HTML and add human-readable date before storing."""
    cleaned = dict(article)
    raw_content = cleaned.get("article_content", "")
    if raw_content:
        cleaned["article_content"] = strip_html(raw_content)
    dt = parse_release_time(cleaned.get("article_release_time"))
    if dt:
        cleaned["date"] = dt.strftime("%Y-%m-%d")
    return cleaned


def _load_archive() -> list[dict]:
    """Load existing archive or return empty list. Backfills missing 'date' field."""
    if ARCHIVE_PATH.exists():
        articles = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
        for a in articles:
            if "date" not in a:
                dt = parse_release_time(a.get("article_release_time"))
                if dt:
                    a["date"] = dt.strftime("%Y-%m-%d")
        return articles
    return []


def _save_archive(articles: list[dict]) -> None:
    """Save archive sorted by date (newest first)."""
    def sort_key(a):
        dt = parse_release_time(a.get("article_release_time"))
        return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

    articles.sort(key=sort_key, reverse=True)

    ARCHIVE_PATH.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_all_available(max_pages: int = 50) -> list[dict]:
    """Fetch all articles currently available from the API (no date filter)."""
    results: list[dict] = []

    for page in range(1, max_pages + 1):
        resp = coinglass_get_raw("/article/list", {"page": page})
        data = resp.get("data")
        if not data or not isinstance(data, list):
            break
        results.extend(data)

        # If page returned fewer than ~20 items, we've hit the end
        if len(data) < 15:
            break

    print(f"[news_collector] Fetched {len(results)} articles from API")
    return results


def collect_news() -> dict:
    """
    Main collection function.
    Fetches all available articles from API, deduplicates against
    existing archive, strips HTML, and saves.

    Returns stats: {before, fetched, new, after, date_range}.
    """
    archive = _load_archive()
    existing_keys = {_make_key(a) for a in archive}
    before_count = len(archive)

    fresh = fetch_all_available()

    # Merge: add only new articles (with HTML stripped)
    new_count = 0
    for article in fresh:
        key = _make_key(article)
        if key not in existing_keys:
            archive.append(_prepare_for_archive(article))
            existing_keys.add(key)
            new_count += 1

    _save_archive(archive)

    # Compute date range in archive
    dates = []
    for a in archive:
        dt = parse_release_time(a.get("article_release_time"))
        if dt:
            dates.append(dt)

    date_range = (
        f"{min(dates).strftime('%Y-%m-%d')} → {max(dates).strftime('%Y-%m-%d')}"
        if dates else "empty"
    )

    stats = {
        "before": before_count,
        "fetched": len(fresh),
        "new": new_count,
        "after": len(archive),
        "date_range": date_range,
    }

    print(f"[news_collector] Archive: {before_count} → {len(archive)} articles (+{new_count} new)")
    print(f"[news_collector] Date range: {date_range}")

    return stats


def get_articles_in_range(
    dt_from: datetime,
    dt_to: datetime,
    fallback_to_api: bool = True,
) -> list[dict]:
    """
    Read articles from the local archive within [dt_from, dt_to].

    If the archive has no articles in this range and fallback_to_api=True,
    fetches directly from the CoinGlass API (limited to ~24 days of history).
    """
    archive = _load_archive()
    results = []
    for article in archive:
        dt = parse_release_time(article.get("article_release_time"))
        if dt and dt_from <= dt <= dt_to:
            results.append(article)

    # Fallback: if archive is empty for this range, try the API directly
    if not results and fallback_to_api:
        print(f"[news_collector] Archive empty for {dt_from.date()} → {dt_to.date()}, falling back to API")
        results = fetch_articles_in_window(dt_from, dt_to)

    results.sort(
        key=lambda a: parse_release_time(a.get("article_release_time")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return results


if __name__ == "__main__":
    stats = collect_news()
    print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")
