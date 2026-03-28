"""
Twitter tweet collector with SQLite archive.

Fetches tweets from configured accounts via Selenium scraper,
deduplicates by tweet_id (via DB constraint), and stores in SQLite.

Usage:
    python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_collector
"""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from MultiagentSystem.agents.twitter_analyser.twitter_classifier import classify_tweets
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twscraper_launcher import (
    create_driver,
    fetch_tweets_sync,
)
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_db import (
    count_tweets,
    delete_tweets_by_ids,
    delete_tweets_in_range,
    get_date_range,
    get_all_tweets,
    insert_tweets,
    update_tweet_signals,
)

ACCOUNTS_CONFIG_PATH = Path(__file__).parent.parent / "twitter_collector_settings.json"
LOG_TAG = "[twitter_collector]"
_MODE_FETCH_NEW = "fetch_new"
_MODE_RECLASSIFY_ALL = "reclassify_all"
_MODE_RECLASSIFY_AND_REFETCH = "reclassify_and_refetch"


def _load_accounts_config() -> dict:
    """Get accounts configuration from JSON file. Convert to dict."""
    return json.loads(ACCOUNTS_CONFIG_PATH.read_text(encoding="utf-8"))


def _is_driver_connection_error(exc: Exception) -> bool:
    """Detect broken Selenium transport/session errors."""
    msg = str(exc).lower()
    markers = (
        "driver_connection_lost",
        "winerror 10054",
        "connection reset",
        "forcibly closed by the remote host",
        "invalid session id",
        "disconnected",
    )
    return any(marker in msg for marker in markers)


def _resolve_cli_mode(argv: list[str]) -> tuple[str | None, str | None]:
    """Resolve collector run mode from CLI flags.

    Returns:
        (mode, error) where exactly one of them is None.
    """
    known = {"--fetch-new", "--reclassify-all", "--reclassify-and-refetch", "--reclassify"}
    unknown = [arg for arg in argv if arg.startswith("--") and arg not in known]
    if unknown:
        return None, f"Unknown flag(s): {', '.join(unknown)}"

    selected: list[str] = []
    if "--fetch-new" in argv:
        selected.append(_MODE_FETCH_NEW)
    if "--reclassify-all" in argv or "--reclassify" in argv:
        selected.append(_MODE_RECLASSIFY_ALL)
    if "--reclassify-and-refetch" in argv:
        selected.append(_MODE_RECLASSIFY_AND_REFETCH)

    if len(selected) > 1:
        return None, (
            "Flags are mutually exclusive. Use exactly one of: "
            "--fetch-new | --reclassify-all | --reclassify-and-refetch"
        )

    if not selected:
        return _MODE_FETCH_NEW, None
    return selected[0], None


def _get_config_date_bounds() -> tuple[str, str]:
    """Read and validate inclusive date bounds from settings."""
    config = _load_accounts_config()
    settings = config.get("scraper_settings", {})
    since_date = settings.get("since_date", "")
    until_date = settings.get("until_date", "")

    if not since_date or not until_date:
        raise ValueError(
            "since_date and until_date must be set in twitter_collector_settings.json"
        )
    if since_date > until_date:
        raise ValueError(f"since_date ({since_date}) > until_date ({until_date})")
    return since_date, until_date


def collect_tweets() -> dict:
    """Main collection function.

    Iterates over configured accounts, fetches tweets with rate limiting,
    filters retweets, and inserts into SQLite (duplicates ignored by DB).

    Returns stats dict with per-account breakdown.
    """
    config = _load_accounts_config()
    # Gather all enabled sources for news
    accounts = [a for a in config["accounts"] if a.get("enabled", True)]
    settings = config["scraper_settings"]

    before_count = count_tweets()

    since_date = settings.get("since_date", "")
    until_date = settings.get("until_date", "")

    if since_date and until_date and since_date > until_date:
        print(
            f"{LOG_TAG} ERROR: since_date ({since_date}) > until_date ({until_date}). "
            "Check twitter_collector_settings.json"
        )
        return {"error": "since_date > until_date", "before": before_count}

    print(
        f"{LOG_TAG} Starting collection: {len(accounts)} accounts, "
        f"date range: [{since_date or 'any'}, {until_date or 'any'}] (inclusive)"
    )

    total_new = 0
    per_account: list[dict] = []

    # Create a single driver for all accounts (CHROME DRIVER WITH COOKIES)
    driver = create_driver(headless=True)

    # Iterate over all accounts and fetch tweets
    try:
        for i, account in enumerate(accounts):
            username = account["username"]
            print(f"{LOG_TAG} [{i + 1}/{len(accounts)}] Fetching @{username}...")

            max_tweets = settings["max_tweets_per_author"]
            max_scrolls = settings.get("max_scrolls", 100)

            account_stat = {
                "username": username,
                "fetched": 0,
                "retweets_filtered": 0,
                "date_filtered": 0,
                "empty_text": 0,
                "original_kept": 0,
                "signal_bull": 0,
                "signal_bear": 0,
                "signal_no_correlation": 0,
                "confidence_high": 0,
                "confidence_middle": 0,
                "confidence_low": 0,
                "signal_filtered_out": 0,
                "new_in_db": 0,
                "error": None,
                "retried_after_driver_restart": False,
            }

            try:
                tweets = fetch_tweets_sync(
                    username,
                    max_tweets=max_tweets if max_tweets > 0 else None,
                    since_date=since_date,
                    until_date=until_date,
                    driver=driver,
                    max_scrolls=max_scrolls,
                )
            except Exception as e:
                if _is_driver_connection_error(e):
                    print(
                        f"{LOG_TAG}   WARNING: driver connection lost on @{username}. "
                        "Restarting browser and retrying once..."
                    )
                    account_stat["retried_after_driver_restart"] = True
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = create_driver(headless=True)
                    try:
                        tweets = fetch_tweets_sync(
                            username,
                            max_tweets=max_tweets if max_tweets > 0 else None,
                            since_date=since_date,
                            until_date=until_date,
                            driver=driver,
                            max_scrolls=max_scrolls,
                        )
                    except Exception as retry_e:
                        print(f"{LOG_TAG}   ERROR fetching @{username} after restart: {retry_e}")
                        account_stat["error"] = str(retry_e)
                        per_account.append(account_stat)
                        continue
                else:
                    print(f"{LOG_TAG}   ERROR fetching @{username}: {e}")
                    account_stat["error"] = str(e)
                    per_account.append(account_stat)
                    continue

            account_stat["fetched"] = len(tweets)

            original_tweets: list[dict] = []
            for t in tweets:
                if t.get("is_retweet"):
                    account_stat["retweets_filtered"] += 1
                    continue
                tweet_date = t.get("date", "")
                if since_date and tweet_date and tweet_date < since_date:
                    account_stat["date_filtered"] += 1
                    continue
                if until_date and tweet_date and tweet_date > until_date:
                    account_stat["date_filtered"] += 1
                    continue
                if not t.get("text", "").strip():
                    account_stat["empty_text"] += 1
                original_tweets.append(t)

            account_stat["original_kept"] = len(original_tweets)

            # Classify all kept tweets before deciding what to persist.
            classify_tweets(original_tweets)

            tweets_to_insert: list[dict] = []
            for t in original_tweets:
                signal = (t.get("signal_type") or "").upper()
                conf = (t.get("signal_confidence") or "").upper()

                if signal == "BULL":
                    account_stat["signal_bull"] += 1
                elif signal == "BEAR":
                    account_stat["signal_bear"] += 1
                else:
                    account_stat["signal_no_correlation"] += 1

                if conf == "HIGH":
                    account_stat["confidence_high"] += 1
                elif conf == "MIDDLE":
                    account_stat["confidence_middle"] += 1
                else:
                    account_stat["confidence_low"] += 1

                # Store only directional BTC signals in DB (BULL/BEAR).
                if signal in {"BULL", "BEAR"}:
                    tweets_to_insert.append(t)
                else:
                    account_stat["signal_filtered_out"] += 1

            new_count = insert_tweets(tweets_to_insert)
            account_stat["new_in_db"] = new_count

            total_new += new_count

            print(
                f"{LOG_TAG}   @{username}: "
                f"{len(tweets)} fetched → "
                f"{account_stat['retweets_filtered']} RT filtered, "
                f"{account_stat['date_filtered']} date filtered, "
                f"{account_stat['empty_text']} empty text, "
                f"{len(original_tweets)} kept, "
                f"signals BULL/BEAR/NC={account_stat['signal_bull']}/"
                f"{account_stat['signal_bear']}/{account_stat['signal_no_correlation']}, "
                f"H/M/L={account_stat['confidence_high']}/"
                f"{account_stat['confidence_middle']}/{account_stat['confidence_low']}, "
                f"{account_stat['signal_filtered_out']} filtered by signal rule → "
                f"{new_count} new in DB"
            )

            if account_stat["empty_text"] > 0:
                print(
                    f"{LOG_TAG}   WARNING: {account_stat['empty_text']} tweets had empty text "
                    f"(still stored — may indicate DOM parsing issue)"
                )

            per_account.append(account_stat)

            if i < len(accounts) - 1:
                pause = random.uniform(settings["pause_min_sec"], settings["pause_max_sec"])
                print(f"{LOG_TAG}   Sleeping {pause:.0f}s (rate limit)...")
                time.sleep(pause)
    finally:
        driver.quit()

    after_count = count_tweets()
    date_range = get_date_range()

    stats = {
        "before": before_count,
        "new": total_new,
        "after": after_count,
        "date_range": date_range,
        "accounts_scraped": len(accounts),
        "since_date": since_date,
        "per_account": per_account,
    }

    print(f"{LOG_TAG} Archive: {before_count} -> {after_count} tweets (+{total_new} new)")
    print(f"{LOG_TAG} Date range: {date_range}")

    return stats


def reclassify_archive() -> dict:
    """Reclassify all stored Twitter rows and keep only BULL/BEAR rows."""
    archive = get_all_tweets()
    before_count = len(archive)
    if not archive:
        print(f"{LOG_TAG} Archive is empty — nothing to reclassify")
        return {"reclassified": 0, "removed": 0, "after": 0}

    print(f"{LOG_TAG} Reclassifying {before_count} archived tweets...")
    # Strict mode prevents destructive mass-deletes on classifier/API failures.
    classify_tweets(archive, force_reclassify=True, strict=True)

    updated = update_tweet_signals(archive)

    # Find all tweets with NEUTRAL
    ids_to_remove: list[str] = []
    for t in archive:
        signal = (t.get("signal_type") or "").upper()
        conf = (t.get("signal_confidence") or "").upper()
        if signal not in {"BULL", "BEAR"}:
            twid = t.get("tweet_id")
            if twid:
                ids_to_remove.append(str(twid))
                
    # Delete neutral tweets
    removed = delete_tweets_by_ids(ids_to_remove)
    after_count = count_tweets()

    result = {
        "reclassified": before_count,
        "updated_signals": updated,
        "removed_no_correlation": removed,
        "after": after_count,
    }
    print(
        f"{LOG_TAG} Reclassify complete: "
        f"{before_count} processed, {removed} removed, {after_count} left"
    )
    return result


def reclassify_and_refetch() -> dict:
    """Reclassify all, clear configured date range, then collect again."""
    rec = reclassify_archive()
    since_date, until_date = _get_config_date_bounds()
    removed_in_range = delete_tweets_in_range(since_date, until_date)
    print(
        f"{LOG_TAG} Cleared date range [{since_date}, {until_date}] (inclusive): "
        f"{removed_in_range} rows removed"
    )
    collected = collect_tweets()
    return {
        "mode": _MODE_RECLASSIFY_AND_REFETCH,
        "since_date": since_date,
        "until_date": until_date,
        "reclassify": rec,
        "removed_in_range": removed_in_range,
        "collect": collected,
    }


if __name__ == "__main__":
    mode, err = _resolve_cli_mode(sys.argv[1:])
    if err:
        print(f"{LOG_TAG} ERROR: {err}")
        print(
            f"{LOG_TAG} Usage: "
            "python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_collector "
            "[--fetch-new | --reclassify-all | --reclassify-and-refetch]"
        )
        raise SystemExit(2)

    if mode == _MODE_RECLASSIFY_ALL:
        # Reclassify all tweets we have in archive
        result = reclassify_archive()
        print(f"\nDone. Result: {json.dumps(result, indent=2)}")
    elif mode == _MODE_RECLASSIFY_AND_REFETCH:
        # Reclassify all tweets and refetch all tweets again in diapazon
        result = reclassify_and_refetch()
        print(f"\nDone. Result: {json.dumps(result, indent=2)}")
    else:
        # Does not touch classified twits
        stats = collect_tweets()
        print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")
