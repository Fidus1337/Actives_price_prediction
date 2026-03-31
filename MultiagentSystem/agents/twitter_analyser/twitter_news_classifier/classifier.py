"""
LLM-based Twitter news classifier for BTC market signals.

Classifies tweets as:
- BEAR
- BULL
- NO_CORRELATION_TO_BTC

and confidence:
- LOW
- MIDDLE
- HIGH

Used by full_scrapping_pipeline.py before writing rows into SQLite.
"""

import json
from typing import Literal, cast
from pathlib import Path
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

LOG_TAG = "[twitter_classifier]"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / "dev.env")

# LangChain structured output can emit noisy pydantic serialization warnings for
# internal `parsed` field metadata; this does not affect classification results.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)


class TweetItem(BaseModel):
    tweet_id: str = Field(description="Stable tweet identifier from input payload")
    signal_type: Literal["BEAR", "BULL", "NO_CORRELATION_TO_BTC"]
    confidence: Literal["LOW", "MIDDLE", "HIGH"]


class TweetClassificationResponse(BaseModel):
    classifications: list[TweetItem]


CLASSIFIER_PROMPT = """\
You are a BTC market-impact classifier for Twitter posts.
Your goal: identify tweets that signal REAL near-term price impact on BTCUSDT.

For each tweet, classify impact on BTCUSDT:
- "BULL": likely positive for BTC price in the next 1-7 days.
- "BEAR": likely negative for BTC price in the next 1-7 days.
- "NO_CORRELATION_TO_BTC": no meaningful BTC price impact.

Confidence scale:
- "HIGH": hard data with direct price catalyst (ETF flows, regulatory decisions,
  large liquidations, major macro events like CPI/FOMC).
- "MIDDLE": meaningful market signal but indirect (on-chain trends,
  funding rate shifts, notable whale moves with clear direction).
- "LOW": weak, speculative, or ambiguous signal.

Critical rules:
1) Distinguish between ANALYTICAL opinions and PROMOTIONAL content:
   - Analytical opinions with specific reasoning, price levels, or data
     references -> classify normally (BULL/BEAR) with LOW or MIDDLE confidence.
   - Cheerleading, hype, and generic promotion without substance
     ("Bitcoin is a gift", "most bullish chart ever", "buy the dip",
     "Bitcoin has been declared dead 470 times") -> NO_CORRELATION_TO_BTC.
2) On-chain transfers: deposits TO exchanges = potential selling pressure (BEAR).
   Withdrawals FROM exchanges = accumulation (BULL).
   Transfers between unknown wallets = NO_CORRELATION_TO_BTC.
3) Historical facts, memes, quotes, and educational content = NO_CORRELATION_TO_BTC.
4) If a tweet contains both bullish and bearish signals, classify by the
   dominant near-term price impact.
5) Be skeptical - most tweets do NOT move markets. When in doubt,
   classify as NO_CORRELATION_TO_BTC.
6) Classify every item. Return exact enums only. Preserve tweet_id exactly.
"""


def _choose_batch_size(total: int) -> int:
    if total < 20:
        return total
    if total <= 60:
        return 20
    return 30


def _prepare_for_classification(tweets: list[dict]) -> list[dict]:
    prepared = []
    for t in tweets:
        text = (t.get("text") or "").strip()
        prepared.append({
            "tweet_id": str(t.get("tweet_id", "")),
            "author": t.get("author_username", ""),
            "date": t.get("date", ""),
            "likes": int(t.get("likes", 0) or 0),
            "retweets": int(t.get("retweets", 0) or 0),
            "replies": int(t.get("replies", 0) or 0),
            "views": int(t.get("views", 0) or 0),
            "text": text[:500],
        })
    return prepared


def _apply_fallback(tweets: list[dict]) -> None:
    for t in tweets:
        if not t.get("signal_type"):
            t["signal_type"] = "NO_CORRELATION_TO_BTC"
            t["signal_confidence"] = "LOW"


def _short_text(text: str, max_len: int = 90) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _log_classification(
    idx: int,
    total: int,
    tweet: dict,
    reason: str,
) -> None:
    tweet_id = str(tweet.get("tweet_id", ""))
    author = str(tweet.get("author_username", ""))
    date = str(tweet.get("date", ""))
    sig = str(tweet.get("signal_type", ""))
    conf = str(tweet.get("signal_confidence", ""))
    text = _short_text(str(tweet.get("text", "")))
    print(
        f"{LOG_TAG} [{idx}/{total}] id={tweet_id} author=@{author} date={date} "
        f"signal={sig} confidence={conf} reason={reason} text='{text}'"
    )


_BTC_RELEVANT_TERMS = (
    "btc", "bitcoin", "$btc", "btcusd", "btcusdt",
    "etf", "spot etf", "sec", "fed", "fomc", "rate cut", "rate hike",
    "inflation", "cpi", "ppi", "treasury", "dollar", "usd liquidity",
    "coinbase", "binance", "strategy", "saylor", "blackrock", "grayscale",
    "whale", "liquidation", "open interest", "funding rate", "risk-on", "risk off",
)

_LIKELY_NEUTRAL_TERMS = (
    "watch the full", "gm", "good morning", "syncing", "thread", "join us",
    "happy", "conference panel", "podcast", "merch", "giveaway", "follow me",
)


def _is_potentially_non_neutral(tweet: dict) -> bool:
    """Heuristic pre-filter: only likely market-relevant tweets go to LLM."""
    text = (tweet.get("text") or "").lower().strip()
    if not text:
        return False

    if any(term in text for term in _BTC_RELEVANT_TERMS):
        return True

    # If no BTC/macro marker and text has a typical social/promo pattern,
    # treat as neutral without spending an LLM call.
    if any(term in text for term in _LIKELY_NEUTRAL_TERMS):
        return False

    # Generic fallback: if text is very short and lacks key terms -> neutral.
    if len(text) < 40:
        return False
    return False


def _group_by_date(items: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        date_key = str(item.get("date") or "unknown")
        groups.setdefault(date_key, []).append(item)
    return sorted(groups.items(), key=lambda x: x[0], reverse=True)


def classify_tweets(
    tweets: list[dict],
    force_reclassify: bool = False,
    strict: bool = False,
) -> None:
    """Classify tweets in-place.

    Adds:
    - signal_type: BEAR/BULL/NO_CORRELATION_TO_BTC
    - signal_confidence: LOW/MIDDLE/HIGH

    Args:
        tweets: tweet dicts to classify in-place.
        force_reclassify: if True, reclassify every item regardless of existing labels.
        strict: if True, raise on LLM init/batch errors (no fallback-to-neutral).
    """
    if not tweets:
        return

    if force_reclassify:
        for t in tweets:
            t.pop("signal_type", None)
            t.pop("signal_confidence", None)

    total = len(tweets)
    idx_by_obj = {id(t): idx for idx, t in enumerate(tweets, start=1)}

    # Empty-text tweets are not useful for signal extraction.
    for idx, t in enumerate(tweets, start=1):
        if not (t.get("text") or "").strip():
            t["signal_type"] = "NO_CORRELATION_TO_BTC"
            t["signal_confidence"] = "LOW"
            _log_classification(idx, total, t, reason="empty_text_fallback")

    # Items that still do not have a valid label after preprocessing.
    unlabeled = [
        t for t in tweets
        if (t.get("signal_type") not in {"BULL", "BEAR", "NO_CORRELATION_TO_BTC"})
    ]
    if not unlabeled:
        return

    # Pre-filter: only potentially non-neutral items are sent to LLM.
    to_classify = []
    for t in unlabeled:
        if _is_potentially_non_neutral(t):
            to_classify.append(t)
        else:
            t["signal_type"] = "NO_CORRELATION_TO_BTC"
            t["signal_confidence"] = "LOW"
            idx = idx_by_obj.get(id(t), 0)
            _log_classification(idx, total, t, reason="prefilter_neutral")

    if not to_classify:
        return

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    except Exception as e:
        msg = f"{LOG_TAG} ERROR init model: {e}"
        print(msg)
        if strict:
            raise RuntimeError(msg) from e
        _apply_fallback(to_classify)
        return

    by_id = {str(t.get("tweet_id", "")): t for t in to_classify}
    idx_by_id = {str(t.get("tweet_id", "")): idx for idx, t in enumerate(tweets, start=1)}
    prepared = _prepare_for_classification(to_classify)
    by_date = _group_by_date(prepared)
    global_batch_id = 0
    running_offset = 0

    for date_key, date_items in by_date:
        batch_size = max(_choose_batch_size(len(date_items)), 1)
        print(
            f"{LOG_TAG} Date {date_key}: {len(date_items)} non-neutral candidates, "
            f"batch_size={batch_size}"
        )
        for i in range(0, len(date_items), batch_size):
            global_batch_id += 1
            batch = date_items[i:i + batch_size]
            print(
                f"{LOG_TAG} Batch {global_batch_id} (date={date_key}): classifying {len(batch)} tweets "
                f"(items {running_offset + i + 1}-{running_offset + i + len(batch)} of {len(prepared)} requiring LLM)"
            )
            try:
                payload = json.dumps(batch, ensure_ascii=False)
                result = cast(
                    TweetClassificationResponse,
                    llm.with_structured_output(TweetClassificationResponse).invoke([
                        SystemMessage(content=CLASSIFIER_PROMPT),
                        HumanMessage(content=(
                            f"Classify these {len(batch)} tweets:\n{payload}"
                        )),
                    ]),
                )
                mapped = {item.tweet_id: item for item in result.classifications}
                for item in batch:
                    tw = by_id.get(item["tweet_id"])
                    if tw is None:
                        continue
                    idx = idx_by_id.get(item["tweet_id"], 0)
                    cls = mapped.get(item["tweet_id"])
                    if cls is None:
                        tw["signal_type"] = "NO_CORRELATION_TO_BTC"
                        tw["signal_confidence"] = "LOW"
                        _log_classification(idx, total, tw, reason="missing_llm_item_fallback")
                    else:
                        tw["signal_type"] = cls.signal_type
                        tw["signal_confidence"] = cls.confidence
                        _log_classification(idx, total, tw, reason="llm")
            except Exception as e:
                print(f"{LOG_TAG} ERROR batch {global_batch_id}: {e}")
                if strict:
                    raise RuntimeError(f"{LOG_TAG} ERROR batch {global_batch_id}: {e}") from e
                for item in batch:
                    tw = by_id.get(item["tweet_id"])
                    if tw is not None:
                        tw["signal_type"] = "NO_CORRELATION_TO_BTC"
                        tw["signal_confidence"] = "LOW"
                        idx = idx_by_id.get(item["tweet_id"], 0)
                        _log_classification(idx, total, tw, reason="batch_error_fallback")
        running_offset += len(date_items)
