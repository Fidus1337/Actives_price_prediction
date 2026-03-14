import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
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
    print(f"{TAG} [STEP 1/6] Settings loaded | horizon={horizon}d | forecast_date={forecast_date} | window={window}d")

    # 2. Date boundaries for filtering
    if isinstance(forecast_date, str):
        dt_to = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt_to = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    dt_from = dt_to - timedelta(days=window)
    print(f"{TAG} [STEP 2/6] Date boundaries: {dt_from.date()} -> {dt_to.date()}")

    # 3. Fetch news from CoinGlass API (with pagination)
    print(f"{TAG} [STEP 3/6] Fetching articles from CoinGlass API...")
    all_articles = fetch_articles_in_window(dt_from, dt_to)
    print(f"{TAG}   Fetched {len(all_articles)} raw articles")

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
    print(f"{TAG} [STEP 4/6] After date filtering: {len(news_for_llm)} articles in window {dt_from.date()} -> {dt_to.date()}")

    # Save for debugging
    (AGENT_DIR / "input_data.json").write_text(
        json.dumps(news_for_llm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{TAG}   Input data saved to input_data.json")

    # If no news — return empty signal
    if not news_for_llm:
        print(f"{TAG}   No news found in window — returning empty signal")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # -- LLM #1: Classification ------------------------------------------------
    print(f"{TAG} [STEP 5/6] Calling classifier LLM to categorize {len(news_for_llm)} articles...")
    news_json = json.dumps(news_for_llm, ensure_ascii=False)
    classifier_llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0.0)

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

    cat_counts = {}
    for art in news_for_llm:
        cat = art.get("btc_correlation", "?")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"{TAG}   Classification results: {cat_counts}")
    print(f"{TAG}   Relevant articles (low+middle+high): {len(relevant_news)} / {len(news_for_llm)}")
    for art in relevant_news:
        print(f"{TAG}     [{art.get('btc_correlation', '?'):>6}] {art['title'][:80]}")

    # If all news is not_correlated — return empty signal
    if not relevant_news:
        print(f"{TAG}   No relevant news found — returning empty signal")
        return {"agent_signals": {"news_analyser_agent": {"summary": None}}}

    # -- LLM #2: Analysis -----------------------------------------------------
    # Load system prompt
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = settings.get("system_prompt", "")

    analyst_llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0.2)

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
        print(f"{TAG}   Including {len(prev_feedback)} previous validator feedback(s)")
    else:
        print(f"{TAG}   No previous validator feedback")

    print(f"{TAG} [STEP 6/6] Calling analyst LLM (gpt-4o-mini) with {len(messages)} messages...")

    response = cast(
        NewsAnalysisResponse,
        analyst_llm.with_structured_output(NewsAnalysisResponse).invoke(messages)
    )

    prediction_label = "HIGHER" if response.prediction else "LOWER"
    print(f"{TAG} LLM response received:")
    print(f"{TAG}   Prediction: {prediction_label}")
    print(f"{TAG}   Confidence: {response.confidence}")
    print(f"{TAG}   Reasoning: {response.reasoning[:200]}...")
    print(f"{TAG}   Summary: {response.summary[:200]}")
    print(f"{TAG}   Risks: {response.risks[:200]}")

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
    print(f"{TAG} news_predict.json saved to {AGENT_DIR}")
    print(f"{TAG} Done. Returning signal to graph.")

    return {"agent_signals": {"news_analyser_agent": {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }}}
