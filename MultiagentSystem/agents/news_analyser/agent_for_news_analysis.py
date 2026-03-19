import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel, Field
from agents.news_analyser.helpers import parse_release_time, strip_html
from agents.news_analyser.news_collector import get_articles_in_range


# -- Pydantic models ----------------------------------------------------------

class NewsItem(BaseModel):
    article_id: str = Field(description="Stable article identifier from input payload")
    title: str
    category: Literal["not_correlated", "bear", "bull"]
    strength: Literal["low", "medium", "high"] = Field(
        description="How strongly this news impacts BTC price. "
        "Only meaningful when category is bear or bull."
    )


class NewsClassificationResponse(BaseModel):
    classifications: list[NewsItem]


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[news_agent]"

# -- Classifier prompt --------------------------------------------------------

CLASSIFIER_PROMPT = """\
You are a crypto-news classifier. For each news article, determine its likely \
impact on the BTCUSDT price.

Categories:
- "not_correlated": news about altcoins, NFTs, specific DeFi protocols, or events \
with no meaningful impact on BTC price.
- "bull": news that is likely POSITIVE for BTC price — institutional adoption, \
ETF approvals, favorable regulation, dovish Fed signals, major partnerships, \
positive on-chain metrics, supply shocks, etc.
- "bear": news that is likely NEGATIVE for BTC price — exchange hacks, \
regulatory crackdowns, hawkish Fed signals, large sell-offs, negative macro data, \
security breaches, bans, bankruptcies, etc.

For each article also estimate the STRENGTH of its impact on BTC price:
- "low": minor or indirect influence, unlikely to move price significantly.
- "medium": notable event that could contribute to a price move.
- "high": major catalyst that can directly drive significant price movement \
(e.g. ETF decision, Fed rate change, large-scale hack, institutional buy).

Return a classification for EVERY article provided.
You MUST copy both fields exactly:
- "article_id" from the input payload
- "title" from the input payload
"""


# -- Pure helpers --------------------------------------------------------------

# Weighted scoring ensures high-impact news has proportional influence over noise
STRENGTH_WEIGHTS = {"low": 1, "medium": 2, "high": 3}


def _choose_batch_size(total_articles: int) -> int:
    """Smaller batches improve LLM classification accuracy;
    single call avoids overhead for small sets."""
    if total_articles < 15:
        return total_articles
    if total_articles <= 30:
        return 20
    return 30


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

    # Neutral point: equal bull and bear weight means no directional signal
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

def _parse_forecast_window(
    forecast_date: str | date | datetime,
    window_days: int,
) -> tuple[datetime, datetime, datetime]:
    """Convert forecast_date + window into (window_start, forecast_end_date, window_end_exclusive).

    Uses exclusive upper bound to avoid double-counting articles published
    exactly at midnight on the boundary date.
    """
    if isinstance(forecast_date, str):
        forecast_end_date = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        forecast_end_date = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    window_end_exclusive = forecast_end_date + timedelta(days=1)
    window_start = window_end_exclusive - timedelta(days=window_days)
    return window_start, forecast_end_date, window_end_exclusive


def _load_articles_in_window(
    window_start: datetime,
    window_end_inclusive: datetime,
    window_end_exclusive: datetime,
) -> list[dict]:
    """Fetch articles from local archive (with API fallback), filter to window,
    and format them for LLM classification."""
    archive = get_articles_in_range(
        dt_from=window_start,
        dt_to=window_end_inclusive,
        fallback_to_api=True,
    )

    articles_to_classify = []
    for idx, item in enumerate(archive):
        release_time = parse_release_time(item.get("article_release_time"))
        if release_time is None or not (window_start <= release_time < window_end_exclusive):
            continue
        articles_to_classify.append({
            "article_id": f"news_{idx}",
            "date": release_time.strftime("%Y-%m-%d %H:%M"),
            "title": item.get("article_title", "—"),
            "source": item.get("source_name", "—"),
            "content": strip_html(item.get("article_content", ""))[:500],
        })

    articles_to_classify.sort(key=lambda x: x["date"], reverse=True)
    return articles_to_classify


def _classify_articles_in_batches(
    articles_to_classify: list[dict],
    batch_size: int,
) -> tuple[list[NewsItem], list[float], list[int]]:
    """Send articles to LLM in adaptive batches.

    Returns (all_classifications, batch_normalized_scores, batch_relevant_sizes).
    batch_normalized_scores/batch_relevant_sizes are only populated when
    total_articles > 60 (per-batch normalization mode).
    """
    classifier_llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0.0)
    total_articles = len(articles_to_classify)

    all_classifications: list[NewsItem] = []
    batch_normalized_scores: list[float] = []
    batch_relevant_sizes: list[int] = []
    effective_batch_size = max(batch_size, 1)

    for batch_start in range(0, total_articles, effective_batch_size):
        batch = articles_to_classify[batch_start:batch_start + effective_batch_size]
        batch_number = batch_start // effective_batch_size + 1
        print(f"{LOG_TAG}   Batch {batch_number}: {len(batch)} articles")

        news_json = json.dumps(batch, ensure_ascii=False)
        classification_result = cast(
            NewsClassificationResponse,
            classifier_llm.with_structured_output(NewsClassificationResponse).invoke([
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(content=(
                    f"Classify these {len(batch)} articles.\n"
                    f"Return fields article_id, title, category, strength for each item:\n{news_json}"
                )),
            ])
        )
        all_classifications.extend(classification_result.classifications)

        # Per-batch normalization prevents a single large batch
        # from dominating the signal when many articles exist
        if total_articles > 60:
            batch_bull_weight = 0.0
            batch_bear_weight = 0.0
            for item in classification_result.classifications:
                weight = STRENGTH_WEIGHTS.get(item.strength, 1)
                if item.category == "bull":
                    batch_bull_weight += weight
                elif item.category == "bear":
                    batch_bear_weight += weight
            total_weight = batch_bull_weight + batch_bear_weight
            if total_weight > 0:
                batch_score = (batch_bull_weight - batch_bear_weight) / total_weight
                batch_normalized_scores.append(batch_score)
                batch_relevant_sizes.append(int(total_weight))

    return all_classifications, batch_normalized_scores, batch_relevant_sizes


def _merge_classifications_with_articles(
    articles_to_classify: list[dict],
    all_classifications: list[NewsItem],
) -> tuple[list[dict], float, float, int, int, int]:
    """Match LLM classifications back to original articles and compute sentiment scores.

    Returns (classified_articles, bull_weight, bear_weight,
             bull_count, bear_count, not_correlated_count).
    """
    # Primary lookup by article_id, with title fallback because
    # LLM sometimes modifies article_id in structured output
    classifications_by_id = {item.article_id: item for item in all_classifications}
    classifications_by_title = {item.title: item for item in all_classifications}

    bull_count = 0
    bear_count = 0
    not_correlated_count = 0
    bull_weight = 0.0
    bear_weight = 0.0
    classified_articles = []
    title_fallback_count = 0

    for article in articles_to_classify:
        matched_classification = classifications_by_id.get(article["article_id"])
        if matched_classification is None:
            matched_classification = classifications_by_title.get(article["title"])
            if matched_classification is not None:
                title_fallback_count += 1

        category = matched_classification.category if matched_classification else "not_correlated"
        strength = matched_classification.strength if matched_classification else "low"
        weight = STRENGTH_WEIGHTS.get(strength, 1)

        article["sentiment"] = category
        article["strength"] = strength
        classified_articles.append(article)

        if category == "bull":
            bull_count += 1
            bull_weight += weight
        elif category == "bear":
            bear_count += 1
            bear_weight += weight
        else:
            not_correlated_count += 1

    if title_fallback_count:
        print(f"{LOG_TAG}   Warning: {title_fallback_count} items matched by title fallback (missing article_id in LLM output)")

    return classified_articles, bull_weight, bear_weight, bull_count, bear_count, not_correlated_count


def _save_input_debug(articles_to_classify: list[dict]) -> None:
    """Debug artifact: raw articles before classification."""
    (AGENT_DIR / "input_data.json").write_text(
        json.dumps(articles_to_classify, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_prediction_debug(
    forecast_date, horizon: int, window_days: int,
    articles_to_classify: list[dict], classified_articles: list[dict],
    batch_size: int, bull_count: int, bull_weight: float,
    bear_count: int, bear_weight: float, not_correlated_count: int,
    bull_ratio: float, is_bullish: bool, confidence: str,
) -> None:
    """Debug artifact for post-mortem analysis of classification quality."""
    news_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window_days,
        "total_articles": len(articles_to_classify),
        "batch_size": batch_size,
        "bull_count": bull_count,
        "bull_weight": bull_weight,
        "bear_count": bear_count,
        "bear_weight": bear_weight,
        "not_correlated_count": not_correlated_count,
        "bull_ratio_weighted": round(bull_ratio, 4),
        "prediction": is_bullish,
        "confidence": confidence,
        "classifications": [
            {
                "article_id": a["article_id"],
                "title": a["title"],
                "sentiment": a.get("sentiment", "?"),
                "strength": a.get("strength", "?"),
            }
            for a in classified_articles
        ],
    }
    (AGENT_DIR / "news_predict.json").write_text(
        json.dumps(news_predict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# -- Main agent function -------------------------------------------------------

def analyze_news_sentiment(state: AgentState):
    """LangGraph node: classify recent crypto news and produce a bull/bear signal."""

    retry_agents = state.get("retry_agents", [])
    my_retries = state.get("retry_counts", {}).get("news_analyser_agent", 0)
    is_first_run = my_retries == 0

    if not is_first_run and "news_analyser_agent" not in retry_agents:
        print(f"{LOG_TAG} Retry not required — skipping (retries so far: {my_retries})")
        return {}

    run_label = "FIRST RUN" if is_first_run else f"RETRY #{my_retries}"
    print(f"\n{'='*60}")
    print(f"{LOG_TAG} === {run_label} ===")
    print(f"{'='*60}")

    # --- Settings ---
    settings = get_agent_settings(state, "agent_for_news_analysis")
    horizon = state["config"]["horizon"]
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    print(f"{LOG_TAG} [1/5] Settings loaded | horizon={horizon}d | forecast_date={forecast_date} | window={window_days}d")

    # --- Date boundaries ---
    window_start, forecast_end_date, window_end_exclusive = _parse_forecast_window(forecast_date, window_days)
    window_end_inclusive = window_end_exclusive - timedelta(microseconds=1)
    print(f"{LOG_TAG} [2/5] Date window: {window_start.date()} -> {forecast_end_date.date()} (inclusive)")

    # --- Load & filter articles ---
    print(f"{LOG_TAG} [3/5] Loading articles from archive/API...")
    articles_to_classify = _load_articles_in_window(window_start, window_end_inclusive, window_end_exclusive)
    print(f"{LOG_TAG}   Found {len(articles_to_classify)} articles in window")

    # _save_input_debug(articles_to_classify)
    # print(f"{LOG_TAG}   input_data.json saved")

    if not articles_to_classify:
        print(f"{LOG_TAG}   No articles found — returning empty signal")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # --- Classify via LLM ---
    total_articles = len(articles_to_classify)
    batch_size = _choose_batch_size(total_articles)
    print(f"{LOG_TAG} [4/5] Classifying {total_articles} articles (batch_size={batch_size})...")

    all_classifications, batch_normalized_scores, batch_relevant_sizes = (
        _classify_articles_in_batches(articles_to_classify, batch_size)
    )

    # --- Match classifications & compute scores ---
    classified_articles, bull_weight, bear_weight, bull_count, bear_count, not_correlated_count = (
        _merge_classifications_with_articles(articles_to_classify, all_classifications)
    )

    print(f"{LOG_TAG}   bull={bull_count} (w={bull_weight}), bear={bear_count} (w={bear_weight}), neutral={not_correlated_count}")
    for article in classified_articles:
        if article["sentiment"] != "not_correlated":
            print(f"{LOG_TAG}     [{article['sentiment']:>4} {article['strength']:>6}] {article['title'][:80]}")

    # --- Compute verdict ---
    # Two aggregation paths: per-batch normalization for >60 articles,
    # global weighted ratio otherwise
    if total_articles > 60 and batch_normalized_scores:
        weighted_sum = sum(s * w for s, w in zip(batch_normalized_scores, batch_relevant_sizes))
        total_relevant_weight = sum(batch_relevant_sizes)
        normalized_score = weighted_sum / total_relevant_weight if total_relevant_weight else 0.0
        is_bullish, confidence, bull_ratio = _compute_verdict_from_normalized_score(normalized_score)
        print(
            f"{LOG_TAG}   Batch-normalized: {len(batch_normalized_scores)} batches, "
            f"score={normalized_score:.3f}"
        )
    else:
        is_bullish, confidence, bull_ratio = _compute_verdict_from_weights(bull_weight, bear_weight)

    prediction_label = "HIGHER" if is_bullish else "LOWER"
    print(f"{LOG_TAG} [5/5] Verdict: {prediction_label} | confidence={confidence} | bull_ratio={bull_ratio:.2f}")

    # --- Save debug output ---
    _save_prediction_debug(
        forecast_date, horizon, window_days,
        articles_to_classify, classified_articles, batch_size,
        bull_count, bull_weight, bear_count, bear_weight, not_correlated_count,
        bull_ratio, is_bullish, confidence,
    )
    print(f"{LOG_TAG}   news_predict.json saved")

    # --- Build agent signal ---
    summary = (
        f"Bull: {bull_count} (w={bull_weight}), Bear: {bear_count} (w={bear_weight}), "
        f"Not correlated: {not_correlated_count}. "
        f"Weighted bull ratio: {bull_ratio:.2f} → {prediction_label}, confidence: {confidence}."
    )
    reasoning = (
        f"Out of {len(articles_to_classify)} articles, {bull_count} are bullish (weight={bull_weight}) "
        f"and {bear_count} are bearish (weight={bear_weight}) "
        f"(+{not_correlated_count} not correlated). "
        f"Weighted bull/bear ratio = {bull_ratio:.2f}."
    )

    return {"agent_signals": {"news_analyser_agent": {
        "reasoning": reasoning,
        "summary": summary,
        "risks": "",
        "prediction": is_bullish,
        "confidence": confidence,
    }}}
