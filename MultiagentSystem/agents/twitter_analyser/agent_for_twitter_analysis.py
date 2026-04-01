import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from multiagent_types import AgentState, get_agent_settings
from agents.twitter_analyser.twitter_scrapper.twitter_db import get_tweets_in_range


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[agent_for_twitter_analysis]"
AGENT_NAME = "agent_for_twitter_analysis"

# Confidence → weight mapping (same scale as news agent)
CONFIDENCE_WEIGHTS = {"HIGH": 3, "MIDDLE": 2, "LOW": 1}


# -- Pure helpers --------------------------------------------------------------

def _distance_to_confidence(distance: float) -> str:
    """Map distance from neutral (0.5) to a confidence label."""
    if distance >= 0.3:
        return "high"
    if distance >= 0.15:
        return "medium"
    return "low"


def _compute_verdict_from_weights(
    bull_weight: float, bear_weight: float,
) -> tuple[bool, str, float]:
    """Compute prediction from absolute weighted scores.

    Returns (is_bullish, confidence, bull_ratio).
    """
    total_weight = bull_weight + bear_weight
    if total_weight == 0:
        return False, "low", 0.5

    bull_ratio = bull_weight / total_weight
    is_bullish = bull_ratio > 0.5
    confidence = _distance_to_confidence(abs(bull_ratio - 0.5))

    return is_bullish, confidence, bull_ratio


def _compute_verdict_from_normalized_score(
    normalized_score: float,
) -> tuple[bool, str, float]:
    """Compute prediction from normalized score in [-1, 1].

    Returns (is_bullish, confidence, bull_ratio_equivalent).
    """
    bull_ratio_equivalent = (normalized_score + 1.0) / 2.0
    is_bullish = normalized_score > 0
    confidence = _distance_to_confidence(abs(normalized_score) / 2.0)
    return is_bullish, confidence, bull_ratio_equivalent


# -- Extracted single-responsibility functions ---------------------------------

def _parse_tweet_created_at(value: str | datetime | None) -> datetime | None:
    """Parse tweet timestamp into timezone-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _compute_time_decay_weight(
    tweet_created_at: str | datetime | None,
    forecast_end_date: datetime,
    half_life_days: float,
) -> tuple[float, float | None]:
    """Compute exponential time-decay weight by tweet age."""
    if half_life_days <= 0:
        return 1.0, None

    created_at = _parse_tweet_created_at(tweet_created_at)
    if created_at is None:
        return 1.0, None

    age_days = max((forecast_end_date - created_at).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (age_days / half_life_days)
    return decay, age_days


def _parse_tweet_utc_day(tweet: dict) -> date | None:
    """Extract tweet UTC day from created_at or fallback date field."""
    created_at = _parse_tweet_created_at(tweet.get("created_at"))
    if created_at is not None:
        return created_at.astimezone(timezone.utc).date()

    date_raw = tweet.get("date")
    if not date_raw:
        return None

    try:
        return datetime.strptime(str(date_raw), "%Y-%m-%d").date()
    except Exception:
        return None


def _tweet_to_signed_score(tweet: dict) -> float | None:
    """Convert tweet direction+confidence into signed points in [-3, 3]."""
    signal_type = (tweet.get("signal_type") or "").upper()
    if signal_type not in {"BULL", "BEAR"}:
        return None

    confidence = (tweet.get("signal_confidence") or "LOW").upper()
    confidence_weight = float(CONFIDENCE_WEIGHTS.get(confidence, 1))
    return confidence_weight if signal_type == "BULL" else -confidence_weight


def _build_author_day_entries(
    tweets: list[dict],
    forecast_end_date: datetime,
    half_life_days: float,
) -> list[dict]:
    """Build one directional aggregate per (author, UTC day)."""
    grouped: dict[tuple[str, str], dict] = {}

    for tweet in tweets:
        signed_score = _tweet_to_signed_score(tweet)
        if signed_score is None:
            continue

        utc_day = _parse_tweet_utc_day(tweet)
        if utc_day is None:
            continue

        author_raw = (tweet.get("author_username") or "").strip()
        author_key = author_raw.lower() or "unknown"
        day_str = utc_day.isoformat()
        key = (author_key, day_str)

        time_decay, _ = _compute_time_decay_weight(
            tweet.get("created_at"), forecast_end_date, half_life_days
        )

        entry = grouped.setdefault(
            key,
            {
                "author": author_raw or "unknown",
                "author_key": author_key,
                "utc_day": day_str,
                "tweets_count": 0,
                "raw_score_sum": 0.0,
                "decay_sum": 0.0,
            },
        )
        entry["tweets_count"] += 1
        entry["raw_score_sum"] += signed_score
        entry["decay_sum"] += time_decay

    author_day_entries: list[dict] = []
    for (_, _), grouped_entry in grouped.items():
        tweets_count = grouped_entry["tweets_count"]
        if tweets_count <= 0:
            continue

        avg_raw_score = grouped_entry["raw_score_sum"] / tweets_count
        avg_decay = grouped_entry["decay_sum"] / tweets_count
        weighted_score = avg_raw_score * avg_decay

        author_day_entries.append(
            {
                "author": grouped_entry["author"],
                "author_key": grouped_entry["author_key"],
                "utc_day": grouped_entry["utc_day"],
                "tweets_count": tweets_count,
                "avg_raw_score": avg_raw_score,
                "avg_decay": avg_decay,
                "weighted_score": weighted_score,
                "bull_weight": max(weighted_score, 0.0),
                "bear_weight": max(-weighted_score, 0.0),
            }
        )

    # Deterministic order: newest day first, then strongest absolute contribution.
    author_day_entries.sort(
        key=lambda e: (e["utc_day"], abs(e["weighted_score"])), reverse=True
    )
    return author_day_entries


def _parse_forecast_window(
    forecast_date: str | date | datetime,
    window_days: int,
) -> tuple[datetime, datetime, datetime]:
    """Convert forecast_date + window into (window_start, forecast_end_date, window_end_exclusive)."""
    if isinstance(forecast_date, str):
        forecast_end_date = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        forecast_end_date = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    window_end_exclusive = forecast_end_date + timedelta(days=1)
    window_start = window_end_exclusive - timedelta(days=window_days)
    return window_start, forecast_end_date, window_end_exclusive


def _load_tweets_in_window(
    window_start: datetime,
    window_end_inclusive: datetime,
    allowed_authors: list[str] | None = None,
) -> list[dict]:
    """Fetch pre-classified tweets from SQLite archive, filter to window.

    Only BULL/BEAR tweets are stored in DB (NO_CORRELATION filtered at collection time).
    If allowed_authors is provided, only tweets from those authors are returned.
    """
    tweets = get_tweets_in_range(dt_from=window_start, dt_to=window_end_inclusive)

    if allowed_authors:
        allowed_lower = {a.lower() for a in allowed_authors}
        tweets = [t for t in tweets if (t.get("author_username") or "").lower() in allowed_lower]

    # Sort newest first
    tweets.sort(key=lambda t: t.get("created_at") or "", reverse=True)

    return tweets


def _aggregate_sentiment(
    tweets: list[dict],
    forecast_end_date: datetime,
    half_life_days: float,
) -> tuple[float, float, int, int, float, int, list[dict]]:
    """Aggregate pre-classified tweets into weighted sentiment scores.

    Returns (
        bull_weight,
        bear_weight,
        bull_count,
        bear_count,
        avg_decay_directional,
        author_day_groups_count,
        author_day_entries,
    ).
    """
    author_day_entries = _build_author_day_entries(
        tweets=tweets, forecast_end_date=forecast_end_date, half_life_days=half_life_days
    )
    bull_weight = sum(e["bull_weight"] for e in author_day_entries)
    bear_weight = sum(e["bear_weight"] for e in author_day_entries)
    bull_count = sum(1 for e in author_day_entries if e["weighted_score"] > 0)
    bear_count = sum(1 for e in author_day_entries if e["weighted_score"] < 0)
    avg_decay = (
        sum(e["avg_decay"] for e in author_day_entries) / len(author_day_entries)
        if author_day_entries
        else 1.0
    )
    return (
        bull_weight,
        bear_weight,
        bull_count,
        bear_count,
        avg_decay,
        len(author_day_entries),
        author_day_entries,
    )


def _compute_batch_normalized_score(
    author_day_entries: list[dict],
    batch_size: int,
) -> float | None:
    """Per-batch normalization for >60 directional author-day groups.

    Returns normalized score in [-1, 1], or None if not applicable.
    """
    total = len(author_day_entries)
    if total <= 60:
        return None

    effective_batch_size = max(batch_size, 1)
    batch_scores: list[float] = []
    batch_weights: list[float] = []

    for batch_start in range(0, total, effective_batch_size):
        batch = author_day_entries[batch_start:batch_start + effective_batch_size]
        batch_bull = 0.0
        batch_bear = 0.0
        for entry in batch:
            batch_bull += float(entry.get("bull_weight", 0.0))
            batch_bear += float(entry.get("bear_weight", 0.0))
        total_w = batch_bull + batch_bear
        if total_w > 0:
            batch_scores.append((batch_bull - batch_bear) / total_w)
            batch_weights.append(total_w)

    if not batch_scores:
        return None

    weighted_sum = sum(s * w for s, w in zip(batch_scores, batch_weights))
    total_relevant = sum(batch_weights)
    return weighted_sum / total_relevant if total_relevant else 0.0


def _save_prediction_debug(
    forecast_date, horizon: int, window_days: int,
    forecast_end_date: datetime, half_life_days: float,
    tweets: list[dict],
    author_day_groups_count: int,
    author_day_entries: list[dict],
    bull_count: int, bull_weight: float,
    bear_count: int, bear_weight: float,
    bull_ratio: float, is_bullish: bool, confidence: str,
    avg_decay_directional: float,
) -> None:
    """Debug artifact for post-mortem analysis of classification quality."""
    twitter_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window_days,
        "half_life_days": half_life_days,
        "total_tweets": len(tweets),
        "total_author_day_groups": author_day_groups_count,
        "bull_count": bull_count,
        "bull_weight_decayed": round(bull_weight, 4),
        "bear_count": bear_count,
        "bear_weight_decayed": round(bear_weight, 4),
        "avg_time_decay_directional": round(avg_decay_directional, 4),
        "bull_ratio_weighted": round(bull_ratio, 4),
        "prediction": is_bullish,
        "confidence": confidence,
        "author_day_aggregates": [
            {
                "author": e["author"],
                "utc_day": e["utc_day"],
                "tweets_count": e["tweets_count"],
                "avg_raw_score": round(e["avg_raw_score"], 6),
                "avg_time_decay": round(e["avg_decay"], 6),
                "weighted_score": round(e["weighted_score"], 6),
                "bull_weight": round(e["bull_weight"], 6),
                "bear_weight": round(e["bear_weight"], 6),
            }
            for e in author_day_entries
        ],
        "classifications": [
            {
                "author": t.get("author_username", "?"),
                "signal": t.get("signal_type", "?"),
                "confidence": t.get("signal_confidence", "?"),
                "date": t.get("date", "?"),
                "age_days": (
                    round(_compute_time_decay_weight(t.get("created_at"), forecast_end_date, half_life_days)[1], 4)
                    if _compute_time_decay_weight(t.get("created_at"), forecast_end_date, half_life_days)[1] is not None
                    else None
                ),
                "time_decay": round(
                    _compute_time_decay_weight(t.get("created_at"), forecast_end_date, half_life_days)[0], 6
                ),
                "text": (t.get("text") or "")[:120],
            }
            for t in tweets
        ],
    }
    (AGENT_DIR / "twitter_predict.json").write_text(
        json.dumps(twitter_predict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# -- Main agent function -------------------------------------------------------

def agent_for_twitter_analysis(state: AgentState):
    """LangGraph node: aggregate pre-classified Twitter signals into a bull/bear verdict.

    Tweets are pre-classified at collection time (see full_scrapping_pipeline.py).
    Only BULL/BEAR tweets are stored in the SQLite archive.
    This node reads classifications and aggregates — no LLM calls.
    """

    retry_agents = state.get("retry_agents", [])
    my_retries = state.get("retry_counts", {}).get(AGENT_NAME, 0)
    is_first_run = my_retries == 0

    if not is_first_run and AGENT_NAME not in retry_agents:
        print(f"{LOG_TAG} Retry not required — skipping (retries so far: {my_retries})")
        return {}

    run_label = "FIRST RUN" if is_first_run else f"RETRY #{my_retries}"
    print(f"\n{'='*60}")
    print(f"{LOG_TAG} === {run_label} ===")
    print(f"{'='*60}")

    # --- Settings ---
    settings = get_agent_settings(state, "agent_for_twitter_analysis")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    half_life_days = float(settings.get("half_life_days", 2.0))
    print(
        f"{LOG_TAG} [1/4] Settings loaded | horizon={horizon}d | forecast_date={forecast_date} "
        f"| window={window_days}d | half_life={half_life_days}d"
    )

    # --- Date boundaries ---
    window_start, forecast_end_date, window_end_exclusive = _parse_forecast_window(forecast_date, window_days)
    window_end_inclusive = window_end_exclusive - timedelta(microseconds=1)
    print(f"{LOG_TAG} [2/4] Date window: {window_start.date()} -> {forecast_end_date.date()} (inclusive)")

    # --- Load pre-classified tweets ---
    allowed_authors = settings.get("authors")
    print(f"{LOG_TAG} [3/4] Loading pre-classified tweets from SQLite archive...")
    if allowed_authors:
        print(f"{LOG_TAG}   Filtering by authors: {allowed_authors}")
    tweets = _load_tweets_in_window(window_start, window_end_inclusive, allowed_authors=allowed_authors)
    print(f"{LOG_TAG}   Found {len(tweets)} tweets in window")

    if not tweets:
        print(f"{LOG_TAG}   No tweets found — returning empty signal")
        return {"agent_signals": {AGENT_NAME: {"summary": None}}}

    # --- Aggregate pre-classified sentiment ---
    (
        bull_weight,
        bear_weight,
        bull_count,
        bear_count,
        avg_decay_directional,
        author_day_groups_count,
        author_day_entries,
    ) = _aggregate_sentiment(
        tweets, forecast_end_date, half_life_days
    )

    print(
        f"{LOG_TAG}   directional author-day groups={author_day_groups_count} | "
        f"bull={bull_count} (decayed_w={bull_weight:.4f}), "
        f"bear={bear_count} (decayed_w={bear_weight:.4f}), avg_decay={avg_decay_directional:.4f}"
    )
    for entry in author_day_entries:
        print(
            f"{LOG_TAG}     @{entry['author']} {entry['utc_day']} | tweets={entry['tweets_count']} "
            f"| avg_raw={entry['avg_raw_score']:.3f} | avg_decay={entry['avg_decay']:.3f} "
            f"| weighted={entry['weighted_score']:.3f}"
        )

    # --- Compute verdict ---
    total_tweets = len(tweets)
    normalized_score = _compute_batch_normalized_score(
        author_day_entries, batch_size=30
    )

    if normalized_score is not None:
        is_bullish, confidence, bull_ratio = _compute_verdict_from_normalized_score(normalized_score)
        print(f"{LOG_TAG}   Batch-normalized: score={normalized_score:.3f}")
    else:
        is_bullish, confidence, bull_ratio = _compute_verdict_from_weights(bull_weight, bear_weight)

    prediction_label = "HIGHER" if is_bullish else "LOWER"
    print(f"{LOG_TAG} [4/4] Verdict: {prediction_label} | confidence={confidence} | bull_ratio={bull_ratio:.2f}")

    # --- Save debug output ---
    _save_prediction_debug(
        forecast_date, horizon, window_days,
        forecast_end_date, half_life_days,
        tweets,
        author_day_groups_count, author_day_entries,
        bull_count, bull_weight, bear_count, bear_weight,
        bull_ratio, is_bullish, confidence,
        avg_decay_directional,
    )
    print(f"{LOG_TAG}   twitter_predict.json saved")

    # --- Build agent signal ---
    summary = (
        f"Bull author-day groups: {bull_count} (w={bull_weight}), "
        f"Bear author-day groups: {bear_count} (w={bear_weight}). "
        f"Weighted bull ratio: {bull_ratio:.2f} → {prediction_label}, confidence: {confidence}."
    )
    reasoning = (
        f"Out of {total_tweets} tweets, {author_day_groups_count} directional author-day groups "
        f"were formed. Bullish groups: {bull_count} (weight={bull_weight}), "
        f"bearish groups: {bear_count} (weight={bear_weight}). "
        f"Weighted bull/bear ratio = {bull_ratio:.2f}."
    )

    return {"agent_signals": {AGENT_NAME: {
        "reasoning": reasoning,
        "summary": summary,
        "risks": "",
        "prediction": is_bullish,
        "confidence": confidence,
    }}}
