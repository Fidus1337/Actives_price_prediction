import json
from pathlib import Path
from typing import cast

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel

AGENT_DIR = Path(__file__).parent
AGENT_NAME = "agent_for_analysing_tech_indicators"

# Agent cannot return None values or the value that does not match the schema
class TechAnalysisResponse(BaseModel):
    reasoning: str        # step-by-step analysis of all indicators following the prompt structure
    summary: str          # brief final conclusion: forecast + confidence + range
    risks: str            # risks and counter-arguments to the forecast (2-5 points or "")
    prediction: bool      # True = HIGHER, False = LOWER (always pick a direction)
    confidence: str       # high / medium / low

def agent_for_analysing_tech_indicators(state: AgentState):
    TAG = "[agent_for_analysing_tech_indicators]"

    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

    # Is it first iteration for this agent?
    retry_agents = state.get("retry_agents", [])
    my_retries = state.get("retry_counts", {}).get(AGENT_NAME, 0)
    is_first_run = my_retries == 0

    if not is_first_run and AGENT_NAME not in retry_agents:
        print(f"{TAG} Retry not required — skipping (retries so far: {my_retries})")
        return {}

    run_label = "FIRST RUN" if is_first_run else f"RETRY #{my_retries}"
    print(f"\n{'='*60}")
    print(f"{TAG} === {run_label} ===")
    print(f"{'='*60}")

    # 1. Get all agent settings
    settings = get_agent_settings(state, "agent_for_analysing_tech_indicators")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    print(f"{TAG} [STEP 1/6] Settings loaded | horizon={horizon}d | forecast_date={forecast_date}")
    print(f"{TAG}   window_to_analysis={settings['window_to_analysis']} | base_feats count={len(settings['base_feats'])}")

    # 2. We should predict values by the forecast_start_date
    df = state["cached_dataset"].copy()
    print(f"{TAG} [STEP 2/6] Cached dataset shape: {df.shape}")
    # We must take only base_feats columns from dataset
    cols = [c for c in settings["base_feats"] if c in df.columns]
    missing_cols = [c for c in settings["base_feats"] if c not in df.columns]
    if missing_cols:
        print(f"{TAG}   WARNING: {len(missing_cols)} features missing from dataset: {missing_cols[:5]}...")
    print(f"{TAG}   Matched {len(cols)}/{len(settings['base_feats'])} features from config")
    # Take the dates before forecast_date (including forecast_date) and take base_feats columns
    df = df.loc[df["date"] <= pd.Timestamp(forecast_date), ["date"] + cols].tail(settings["window_to_analysis"])
    print(f"{TAG}   Filtered data shape: {df.shape} | date range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

    # Validate that the last date in data matches forecast_date
    last_date = df["date"].iloc[-1].date() if len(df) > 0 else None
    if last_date != forecast_date:
        raise ValueError(
            f"{TAG} Data ends at {last_date}, expected {forecast_date}. "
            f"SharedBaseDataCache may have a data gap."
        )
    print(f"{TAG}   Last date validated: {last_date} == {forecast_date}")

    # 3. Convert to JSON for the prompt and save for debugging
    data_json = df.to_json(orient="records", date_format="iso")
    (AGENT_DIR / "input_data.json").write_text(data_json, encoding="utf-8")
    print(f"{TAG} [STEP 3/6] Input data saved to input_data.json ({len(data_json)} chars)")

    # 4. Extract current closing price (last row)
    close_col = "spot_price_history__close"
    close_price = df[close_col].iloc[-1] if close_col in df.columns else "N/A"
    print(f"{TAG} [STEP 4/6] Close price on {forecast_date}: {close_price}")
    if close_price == "N/A":
        msg = f"Cannot make prediction without close_price for date {forecast_date}"
        print(f"{TAG}   ERROR: {msg}")
        return {"agent_signals": {AGENT_NAME: {
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
        print(f"{TAG} [STEP 5/6] System prompt loaded from file: {settings['system_prompt_file']} ({len(system_prompt)} chars)")
    else:
        system_prompt = settings["system_prompt"]
        print(f"{TAG} [STEP 5/6] System prompt loaded from config ({len(system_prompt)} chars)")

    # 6. Call LLM with CoT: reasoning is filled first, summary is based on it
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prev_feedback: list[str] = (
        state.get("agent_signals", {})
        .get(AGENT_NAME, {})
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
        print(f"{TAG}   Including {len(prev_feedback)} previous validator feedback(s)")
    else:
        print(f"{TAG}   No previous validator feedback")

    print(f"{TAG} [STEP 6/6] Calling LLM (gpt-4o-mini) with {len(messages)} messages...")

    # Tells the LLM to return a JSON object that matches the Pydantic schema, instead of free-form text.
    response = cast(TechAnalysisResponse, llm.with_structured_output(TechAnalysisResponse).invoke(messages))

    prediction_label = "HIGHER" if response.prediction else "LOWER"
    print(f"{TAG} LLM response received:")
    print(f"{TAG}   Prediction: {prediction_label}")
    print(f"{TAG}   Confidence: {response.confidence}")
    print(f"{TAG}   Reasoning: {response.reasoning[:200]}...")
    print(f"{TAG}   Summary: {response.summary[:200]}")
    print(f"{TAG}   Risks: {response.risks[:200]}")

    print(f"{TAG} Done. Returning signal to graph.")
    return {"agent_signals": {AGENT_NAME: {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }}}
