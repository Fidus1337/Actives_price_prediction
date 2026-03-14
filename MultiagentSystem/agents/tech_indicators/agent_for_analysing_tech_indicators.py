import json
from pathlib import Path
from typing import cast

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel

AGENT_DIR = Path(__file__).parent

# Agent cannot return None values or the value that does not match the schema
class TechAnalysisResponse(BaseModel):
    reasoning: str        # step-by-step analysis of all indicators following the prompt structure
    summary: str          # brief final conclusion: forecast + confidence + range
    risks: str            # risks and counter-arguments to the forecast (2-5 points or "")
    prediction: bool      # True = HIGHER, False = LOWER (always pick a direction)
    confidence: str       # high / medium / low

def agent_a_tech(state: AgentState):
    # Is it first iteration for this agent?
    retry_agents = state.get("retry_agents", [])
    is_first_run = state.get("retry_count", 0) <= 1

    if not is_first_run and "tech_analyser_agent" not in retry_agents:
        print("[agent_a_tech] retry not required — skipping")
        return {}

    print("[agent_a_tech] Starting technical indicators analysis...")

    # 1. Get all agent settings
    settings = get_agent_settings(state, "agent_for_analysing_tech_indicators")
    horizon = state["config"]["horizon"]
    forecast_date = state["forecast_start_date"]

    # 2. We should predict values by the forecast_start_date
    df = state["cached_dataset"].copy()
    # We must take only base_feats columns from dataset
    cols = [c for c in settings["base_feats"] if c in df.columns]
    # Take the dates before forecast_date (including forecast_date) and take base_feats columns
    df = df.loc[df["date"] <= pd.Timestamp(forecast_date), ["date"] + cols].tail(settings["window_to_analysis"])
    print(df)

    # Validate that the last date in data matches forecast_date
    last_date = df["date"].iloc[-1].date() if len(df) > 0 else None
    if last_date != forecast_date:
        raise ValueError(
            f"[agent_tech] Data ends at {last_date}, expected {forecast_date}. "
            f"SharedBaseDataCache may have a data gap."
        )

    # 3. Convert to JSON for the prompt and save for debugging
    data_json = df.to_json(orient="records", date_format="iso")
    (AGENT_DIR / "input_data.json").write_text(data_json, encoding="utf-8")

    # 4. Extract current closing price (last row)
    close_col = "spot_price_history__close"
    close_price = df[close_col].iloc[-1] if close_col in df.columns else "N/A"
    if close_price == "N/A":
        msg = f"Cannot make prediction without close_price for date {forecast_date}"
        print(f"[agent_a_tech] {msg}")
        return {"agent_signals": {"tech_analyser_agent": {
            "reasoning": msg,
            "summary": msg,
            "risks": "",
            "prediction": False,
            "confidence": "low",
        }}}

    # 5. Load system prompt (from file or inline)
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = settings["system_prompt"]

    # 6. Call LLM with CoT: reasoning is filled first, summary is based on it
    llm = ChatOpenAI(model="gpt-5.1", temperature=0.2)

    prev_feedback: list[str] = (
        state.get("agent_signals", {})
        .get("tech_analyser_agent", {})
        .get("description_of_the_reports_problem", [])
    )

    # 6. Build conversation prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Data for {settings['window_to_analysis']} days up to {forecast_date} inclusive:\n{data_json}\n\n"
            f"Current BTC closing price: {close_price}\n"
            f"Forecast horizon: {horizon} days\n"
            f"Forecast date: {forecast_date}\n\n"
            f"Perform analysis following the three steps from the instructions and provide a forecast."
        )),
    ]
    
    # Add previous feedbacks
    if prev_feedback:
        history_text = "\n".join(
            f"Iteration {i+1}: {d}" for i, d in enumerate(prev_feedback)
        )
        messages.append(HumanMessage(content=(
            f"VALIDATOR FEEDBACK ON PREVIOUS REPORT VERSIONS:\n{history_text}\n\n"
            f"Take this feedback into account when composing the new report."
        )))

    # Tells the LLM to return a JSON object that matches the Pydantic schema, instead of free-form text.
    response = cast(TechAnalysisResponse, llm.with_structured_output(TechAnalysisResponse).invoke(messages))

    prediction_label = "HIGHER" if response.prediction else "LOWER"
    print(f"[agent_a_tech] Done. Prediction: {prediction_label} | summary: {response.summary[:120]}...")

    return {"agent_signals": {"tech_analyser_agent": {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }}}
