import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel, Field
from agents.news_analyser.helpers import fetch_articles_in_window, parse_release_time, strip_html


# -- Pydantic models ----------------------------------------------------------

class NewsItem(BaseModel):
    title: str
    category: str = Field(description="not_correlated | low | middle | high")


class NewsClassificationResponse(BaseModel):
    classifications: list[NewsItem]


class NewsAnalysisResponse(BaseModel):
    reasoning: str
    summary: str
    risks: str
    prediction: bool      # True = HIGHER, False = LOWER
    confidence: str       # high / medium / low


AGENT_DIR = Path(__file__).parent

# -- Classifier prompt (LLM #1) -----------------------------------------------

CLASSIFIER_PROMPT = """\
You are a crypto-news classifier. For each news article, determine how strongly \
it correlates with the BTCUSDT price movement.

Categories:
- "not_correlated": news about altcoins, NFTs, specific DeFi protocols, or events \
with no meaningful impact on BTC price.
- "low": indirectly related — general crypto market sentiment, minor exchange news, \
small regulatory updates in non-major countries.
- "middle": moderately related — major exchange events (Binance, Coinbase), \
significant regulatory actions (SEC, EU), macroeconomic news that affects risk assets.
- "high": directly impacts BTC — Bitcoin ETF decisions, Fed rate decisions, \
BTC-specific on-chain events, major institutional adoption, BTC halving-related news.

Return a classification for EVERY article provided. Use the exact article title \
in the "title" field.
"""


# -- Main agent function -------------------------------------------------------

def agent_c_news(state: AgentState):
    retry_agents = state.get("retry_agents", [])
    is_first_run = state.get("retry_count", 0) <= 1
    if not is_first_run and "news_analyser_agent" not in retry_agents:
        print("[agent_c_news] retry not required — skipping")
        return {}

    print("[agent_c_news] Starting news analysis...")

    # 1. Settings from config
    settings = get_agent_settings(state, "agent_for_news_analysis")
    horizon = state["config"]["horizon"]
    forecast_date = state["forecast_start_date"]
    window = settings["window_to_analysis"]

    # 2. Date boundaries for filtering
    if isinstance(forecast_date, str):
        dt_to = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt_to = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    dt_from = dt_to - timedelta(days=window)

    # 3. Fetch news from CoinGlass API (with pagination)
    all_articles = fetch_articles_in_window(dt_from, dt_to)
    print(f"[agent_c_news] Fetched {len(all_articles)} articles in window")

    # 4. Filter by date and strip HTML
    news_for_llm = []
    for item in all_articles:
        dt = parse_release_time(item.get("article_release_time"))
        if dt is None or not (dt_from <= dt <= dt_to):
            continue
        news_for_llm.append({
            "date": dt.strftime("%Y-%m-%d %H:%M"),
            "title": item.get("article_title", "—"),
            "source": item.get("source_name", "—"),
            "content": strip_html(item.get("article_content", ""))[:500],
        })

    # Sort by date (newest first)
    news_for_llm.sort(key=lambda x: x["date"], reverse=True)
    print(f"[agent_c_news] Window {dt_from.date()} -> {dt_to.date()}: {len(news_for_llm)} articles")

    # Save for debugging
    (AGENT_DIR / "input_data.json").write_text(
        json.dumps(news_for_llm, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # If no news — return empty signal
    if not news_for_llm:
        print("[agent_c_news] No news in window — skipping")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # -- LLM #1: Classification ------------------------------------------------
    news_json = json.dumps(news_for_llm, ensure_ascii=False)
    classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    clf_response = cast(
        NewsClassificationResponse,
        classifier_llm.with_structured_output(NewsClassificationResponse).invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=f"Classify these {len(news_for_llm)} articles:\n{news_json}"),
        ])
    )

    # Map title -> category
    category_map = {item.title: item.category for item in clf_response.classifications}

    # Filter: keep low + middle + high
    relevant_news = []
    for article in news_for_llm:
        cat = category_map.get(article["title"], "not_correlated")
        article["btc_correlation"] = cat
        if cat != "not_correlated":
            relevant_news.append(article)

    print(f"[agent_c_news] Classification: {len(relevant_news)} relevant out of {len(news_for_llm)}")
    for art in news_for_llm:
        print(f"  [{art.get('btc_correlation', '?'):>15}] {art['title'][:80]}")

    # If all news is not_correlated — return empty signal
    if not relevant_news:
        print("[agent_c_news] No relevant news — skipping")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # -- LLM #2: Analysis -----------------------------------------------------
    # Load system prompt
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = settings.get("system_prompt", "")

    analyst_llm = ChatOpenAI(model="gpt-5.1", temperature=0.2)

    relevant_json = json.dumps(relevant_news, ensure_ascii=False)

    prev_feedback: list[str] = (
        state.get("agent_signals", {})
        .get("news_analyser_agent", {})
        .get("description_of_the_reports_problem", [])
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Relevant news for {window} days up to {forecast_date} "
            f"(filtered by BTC correlation):\n{relevant_json}\n\n"
            f"Forecast horizon: {horizon} days\n"
            f"Forecast date: {forecast_date}\n\n"
            f"Analyze the news and provide a forecast."
        )),
    ]

    if prev_feedback:
        history_text = "\n".join(
            f"Iteration {i+1}: {d}" for i, d in enumerate(prev_feedback)
        )
        messages.append(HumanMessage(content=(
            f"VALIDATOR FEEDBACK ON PREVIOUS REPORT VERSIONS:\n{history_text}\n\n"
            f"Take this feedback into account when composing the new report."
        )))

    response = cast(
        NewsAnalysisResponse,
        analyst_llm.with_structured_output(NewsAnalysisResponse).invoke(messages)
    )

    prediction_label = "HIGHER" if response.prediction else "LOWER"
    print(f"[agent_c_news] Done. Prediction: {prediction_label} | summary: {response.summary[:120]}...")

    # Save debug JSON
    news_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window,
        "total_articles": len(news_for_llm),
        "relevant_articles": len(relevant_news),
        "classifications": [
            {"title": a["title"], "category": a.get("btc_correlation", "?")}
            for a in news_for_llm
        ],
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }
    (AGENT_DIR / "news_predict.json").write_text(
        json.dumps(news_predict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[agent_c_news] news_predict.json saved to {AGENT_DIR}")

    return {"agent_signals": {"news_analyser_agent": {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }}}
