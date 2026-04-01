"""
LangGraph node: analyze macro-economic calendar events for BTC signal.

Unlike the news agent, there is NO pre-classification step.
All filtered events (Major all countries + Medium US only) are sent
to the LLM in a single prompt. The LLM aggregates and returns a verdict.

Flow:
    calendar_archive → filter → format prompt → LLM → AgentSignal
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from multiagent_types import AgentState, get_agent_settings
from .calendar_collector import get_events_in_range


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[agent_for_economic_calendar_analysis]"
AGENT_NAME = "agent_for_economic_calendar_analysis"

class CalendarVerdict(BaseModel):
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="2-3 sentences explaining which events dominate and why")


# -- Helpers -------------------------------------------------------------------

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


def _filter_events_by_importance(events: list[dict]) -> list[dict]:
    """Major (imp>=3) all countries only.
    Exclude 'Waiting' and events without published_value."""
    filtered = []
    for e in events:
        if not e.get("published_value") or e.get("data_effect") == "Waiting":
            continue
        if e.get("importance_level", 0) >= 3:
            filtered.append(e)
    return filtered


def _format_event(event: dict) -> str:
    """Format a single event as a compact string for the LLM prompt."""
    name = event.get("calendar_name", "?")
    country = event.get("country_name", "?")
    imp = "Major" if event.get("importance_level", 0) >= 3 else "Medium"
    actual = event.get("published_value", "—")
    forecast = event.get("forecast_value", "") or "—"
    previous = event.get("previous_value", "") or "—"
    effect = event.get("data_effect", "—")
    dt = event.get("date", "?")
    return (
        f"[{imp}] {dt} [{country}] {name}\n"
        f"  actual: {actual} | forecast: {forecast} | previous: {previous}\n"
        f"  data_effect: {effect}"
    )


def _format_all_events(events: list[dict]) -> str:
    """Format all events into a single text block."""
    return "\n\n".join(_format_event(e) for e in events)


def _save_prediction_debug(
    forecast_date,
    horizon: int,
    window_days: int,
    events: list[dict],
    verdict: CalendarVerdict | None,
) -> None:
    """Save debug artifact for post-mortem analysis."""
    debug = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window_days,
        "total_events": len(events),
        "events": [
            {
                "date": e.get("date", "?"),
                "name": e.get("calendar_name", "?"),
                "country": e.get("country_name", "?"),
                "importance": e.get("importance_level", 0),
                "actual": e.get("published_value", ""),
                "forecast": e.get("forecast_value", ""),
                "previous": e.get("previous_value", ""),
                "data_effect": e.get("data_effect", ""),
            }
            for e in events
        ],
        "verdict": {
            "direction": verdict.direction,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
        } if verdict else None,
    }
    (AGENT_DIR / "calendar_predict.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def get_system_prompt()-> str:
    """Getting the prompt txt, which we have in the folder where lay the file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "system_prompt.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text



# -- Main agent function -------------------------------------------------------

def agent_for_economic_calendar_analysis(state: AgentState):
    """LangGraph node: analyze macro-economic calendar events for BTC signal.

    All filtered events are sent to the LLM in a single prompt.
    No pre-classification — the LLM receives raw events and decides.
    """

    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{LOG_TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

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
    settings = get_agent_settings(state, "agent_for_economic_calendar_analysis")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    print(f"{LOG_TAG} [1/4] Settings | horizon={horizon}d | forecast_date={forecast_date} | window={window_days}d")

    # --- Date boundaries ---
    window_start, forecast_end_date, window_end_exclusive = _parse_forecast_window(forecast_date, window_days)
    window_end_inclusive = window_end_exclusive - timedelta(microseconds=1)
    print(f"{LOG_TAG} [2/4] Date window: {window_start.date()} -> {forecast_end_date.date()} (inclusive)")

    # --- Load and filter events ---
    print(f"{LOG_TAG} [3/4] Loading events from archive...")
    all_events = get_events_in_range(dt_from=window_start, dt_to=window_end_inclusive)
    filtered = _filter_events_by_importance(all_events)
    print(f"{LOG_TAG}   Raw: {len(all_events)} events | After filter (Major only): {len(filtered)}")

    if not filtered:
        print(f"{LOG_TAG}   No events found — returning empty signal")
        return {"agent_signals": {AGENT_NAME: {"summary": None}}}

    # --- LLM call ---
    events_text = _format_all_events(filtered)
    SYSTEM_PROMPT = get_system_prompt()
    system_msg = SYSTEM_PROMPT.format(horizon=horizon, window=window_days)

    print(f"{LOG_TAG}   Sending {len(filtered)} events to LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    try:
        verdict = llm.with_structured_output(CalendarVerdict).invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Analyze these {len(filtered)} economic calendar events:\n\n{events_text}"),
        ])
    except Exception as exc:
        err = f"LLM request failed in {AGENT_NAME}: {exc}"
        print(f"{LOG_TAG}   ERROR: {err}")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": err,
            "summary": "LLM temporarily unavailable. Fallback signal returned.",
            "risks": "Network/API issue during model call.",
            "prediction": False,
            "confidence": "low",
        }}}

    is_bullish = verdict.direction == "bullish"
    prediction_label = "HIGHER" if is_bullish else "LOWER"
    print(f"{LOG_TAG} [4/4] Verdict: {verdict.direction} | confidence={verdict.confidence} | → {prediction_label}")
    print(f"{LOG_TAG}   Reasoning: {verdict.reasoning}")

    # --- Save debug output ---
    _save_prediction_debug(forecast_date, horizon, window_days, filtered, verdict)
    print(f"{LOG_TAG}   calendar_predict.json saved")

    # --- Build agent signal ---
    summary = (
        f"{len(filtered)} macro events analyzed. "
        f"LLM verdict: {verdict.direction}, confidence: {verdict.confidence}."
    )

    return {"agent_signals": {AGENT_NAME: {
        "reasoning": verdict.reasoning,
        "summary": summary,
        "risks": "",
        "prediction": is_bullish,
        "confidence": verdict.confidence,
    }}}
