import json
from datetime import datetime, timedelta, timezone
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


# -- Helpers ------------------------------------------------------------------

STRENGTH_WEIGHTS = {"low": 1, "medium": 2, "high": 3}


def _compute_verdict(
    bull_weight: float, bear_weight: float,
) -> tuple[bool, str, float]:
    """Compute prediction, confidence and bull_ratio from weighted scores.

    Returns (prediction, confidence, bull_ratio).
    """
    total = bull_weight + bear_weight
    if total == 0:
        return False, "low", 0.5

    bull_ratio = bull_weight / total
    prediction = bull_ratio > 0.5
    distance = abs(bull_ratio - 0.5)

    if distance >= 0.3:
        confidence = "high"
    elif distance >= 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    return prediction, confidence, bull_ratio


# -- Main agent function -------------------------------------------------------

def agent_c_news(state: AgentState):
    TAG = "[agent_c_news]"

    retry_agents = state.get("retry_agents", [])
    my_retries = state.get("retry_counts", {}).get("news_analyser_agent", 0)
    is_first_run = my_retries == 0
    if not is_first_run and "news_analyser_agent" not in retry_agents:
        print(f"{TAG} Retry not required — skipping (retries so far: {my_retries})")
        return {}

    run_label = "FIRST RUN" if is_first_run else f"RETRY #{my_retries}"
    print(f"\n{'='*60}")
    print(f"{TAG} === {run_label} ===")
    print(f"{'='*60}")

    # 1. Settings from config
    settings = get_agent_settings(state, "agent_for_news_analysis")
    horizon = state["config"]["horizon"]
    forecast_date = state["forecast_start_date"]
    window = settings["window_to_analysis"]
    print(f"{TAG} [STEP 1/5] Settings loaded | horizon={horizon}d | forecast_date={forecast_date} | window={window}d")

    # 2. Date boundaries for filtering
    if isinstance(forecast_date, str):
        dt_end_day = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt_end_day = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    dt_to_exclusive = dt_end_day + timedelta(days=1)
    dt_to_inclusive = dt_to_exclusive - timedelta(microseconds=1)
    dt_from = dt_to_exclusive - timedelta(days=window)
    print(f"{TAG} [STEP 2/5] Date boundaries: {dt_from.date()} -> {dt_end_day.date()} (inclusive)")

    # 3. Load news from local archive (with API fallback)
    print(f"{TAG} [STEP 3/5] Loading articles from archive/API...")
    archive = get_articles_in_range(
        dt_from=dt_from,
        dt_to=dt_to_inclusive,
        fallback_to_api=True,
    )

    news_for_llm = []
    for idx, item in enumerate(archive):
        dt = parse_release_time(item.get("article_release_time"))
        if dt is None or not (dt_from <= dt < dt_to_exclusive):
            continue
        news_for_llm.append({
            "article_id": f"news_{idx}",
            "date": dt.strftime("%Y-%m-%d %H:%M"),
            "title": item.get("article_title", "—"),
            "source": item.get("source_name", "—"),
            "content": strip_html(item.get("article_content", ""))[:500],
        })

    news_for_llm.sort(key=lambda x: x["date"], reverse=True)
    print(f"{TAG}   Found {len(news_for_llm)} articles in window {dt_from.date()} -> {dt_end_day.date()}")

    # Save for debugging
    (AGENT_DIR / "input_data.json").write_text(
        json.dumps(news_for_llm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{TAG}   Input data saved to input_data.json")

    # If no news — return empty signal
    if not news_for_llm:
        print(f"{TAG}   No news found in window — returning empty signal")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # 4. LLM: Classify each article as bull / bear / not_correlated
    print(f"{TAG} [STEP 4/5] Calling classifier LLM for {len(news_for_llm)} articles...")
    news_json = json.dumps(news_for_llm, ensure_ascii=False)
    classifier_llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0.0)

    clf_response = cast(
        NewsClassificationResponse,
        classifier_llm.with_structured_output(NewsClassificationResponse).invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=(
                f"Classify these {len(news_for_llm)} articles.\n"
                f"Return fields article_id, title, category for each item:\n{news_json}"
            )),
        ])
    )

    # Prefer stable article_id mapping, with title fallback for partial responses.
    clf_by_id = {item.article_id: item for item in clf_response.classifications}
    clf_by_title = {item.title: item for item in clf_response.classifications}

    bull_count = 0
    bear_count = 0
    not_correlated_count = 0
    bull_weight = 0.0
    bear_weight = 0.0
    classified_articles = []
    fallback_matches = 0

    for article in news_for_llm:
        clf_item = clf_by_id.get(article["article_id"])
        if clf_item is None:
            clf_item = clf_by_title.get(article["title"])
            if clf_item is not None:
                fallback_matches += 1

        cat = clf_item.category if clf_item else "not_correlated"
        strength = clf_item.strength if clf_item else "low"
        w = STRENGTH_WEIGHTS.get(strength, 1)

        article["sentiment"] = cat
        article["strength"] = strength
        classified_articles.append(article)

        if cat == "bull":
            bull_count += 1
            bull_weight += w
        elif cat == "bear":
            bear_count += 1
            bear_weight += w
        else:
            not_correlated_count += 1

    if fallback_matches:
        print(f"{TAG}   Warning: {fallback_matches} items matched by title fallback (missing article_id in LLM output)")

    print(f"{TAG}   Classification: bull={bull_count} (weight={bull_weight}), bear={bear_count} (weight={bear_weight}), not_correlated={not_correlated_count}")
    for art in classified_articles:
        if art["sentiment"] != "not_correlated":
            print(f"{TAG}     [{art['sentiment']:>4} {art['strength']:>6}] {art['title'][:80]}")

    # 5. Compute verdict from weighted bear/bull ratio
    prediction, confidence, bull_ratio = _compute_verdict(bull_weight, bear_weight)
    prediction_label = "HIGHER" if prediction else "LOWER"

    summary = (
        f"Bull: {bull_count} (w={bull_weight}), Bear: {bear_count} (w={bear_weight}), "
        f"Not correlated: {not_correlated_count}. "
        f"Weighted bull ratio: {bull_ratio:.2f} → {prediction_label}, confidence: {confidence}."
    )
    reasoning = (
        f"Out of {len(news_for_llm)} articles, {bull_count} are bullish (weight={bull_weight}) "
        f"and {bear_count} are bearish (weight={bear_weight}) "
        f"(+{not_correlated_count} not correlated). "
        f"Weighted bull/bear ratio = {bull_ratio:.2f}."
    )

    print(f"{TAG} [STEP 5/5] Verdict: {prediction_label} | confidence={confidence} | bull_ratio={bull_ratio:.2f}")

    # Save debug JSON
    news_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window,
        "total_articles": len(news_for_llm),
        "bull_count": bull_count,
        "bull_weight": bull_weight,
        "bear_count": bear_count,
        "bear_weight": bear_weight,
        "not_correlated_count": not_correlated_count,
        "bull_ratio_weighted": round(bull_ratio, 4),
        "prediction": prediction,
        "confidence": confidence,
        "classifications": [
            {"article_id": a["article_id"], "title": a["title"], "sentiment": a.get("sentiment", "?"), "strength": a.get("strength", "?")}
            for a in classified_articles
        ],
    }
    (AGENT_DIR / "news_predict.json").write_text(
        json.dumps(news_predict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{TAG} news_predict.json saved to {AGENT_DIR}")
    print(f"{TAG} Done. Returning signal to graph.")

    return {"agent_signals": {"news_analyser_agent": {
        "reasoning": reasoning,
        "summary": summary,
        "risks": "",
        "prediction": prediction,
        "confidence": confidence,
    }}}
