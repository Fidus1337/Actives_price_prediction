import json
import os
import sys
from pathlib import Path
from .multiagent_predictions_module import make_prediction_for_last_N_days, add_y_true, build_confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

from langgraph.graph import StateGraph, START, END

from multiagent_types import AgentState, AgentRetry

from .agents.twitter_analyser.agent_for_twitter_analysis import agent_for_twitter_analysis

from .agents.tech_indicators import agent_for_analysing_tech_indicators

from .agents.verdicts_validator import agent_for_verdicts_validation
from .agents.reports_analyser import agent_reports_analyser

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# News agents are excluded from retry tracking (formula-based, no LLM recompose)
_NEWS_AGENTS = frozenset({"agent_for_twitter_analysis"})

# Max retries per agent (default 2)
MAX_RETRIES = 2

# Node for starting analysis by agents
def supervisor_node(state: AgentState):
    retry_agents: list[AgentRetry] = state.get("retry_agents", [])

    if not retry_agents:
        # First run: initialize retry tracking for every non-news agent involved
        involved: list[str] = state.get("agent_envolved_in_prediction", [])
        initialized: list[AgentRetry] = [
            AgentRetry(
                agent_name=name,
                max_retries=MAX_RETRIES,
                currents_retry=0,
                retry_requirements=[],
            )
            for name in involved
            if name not in _NEWS_AGENTS
        ]
        names = [r["agent_name"] for r in initialized]
        print(f"\n[supervisor] First run — retry tracking initialized for: {names}")
        return {"retry_agents": initialized}

    names = [r["agent_name"] for r in retry_agents if r.get("retry_requirements")]
    print(f"\n[supervisor] Retry run — agents with requirements: {names}")
    return {}


# Router for retry (look schema at miro)
def _should_retry(state: AgentState) -> str:
    retry_agents: list[AgentRetry] = state.get("retry_agents", [])

    # If the agent has any problems - retry
    agents_with_budget = [
        r for r in retry_agents
        if r.get("retry_requirements") and len(r["retry_requirements"]) < r["max_retries"]
    ]

    if agents_with_budget:
        names = [r["agent_name"] for r in agents_with_budget]
        print(f"\n[router] Agents need retry: {names}")
        return "supervisor"

    exhausted = [
        r["agent_name"] for r in retry_agents
        if r.get("retry_requirements") and len(r["retry_requirements"]) >= r["max_retries"]
    ]
    if exhausted:
        print(f"\n[router] Retry limit ({MAX_RETRIES}) exhausted for: {exhausted}")
    else:
        print(f"\n[router] All agents passed validation — finishing.")

    return "agent_reports_analyser"


# ==========================================
# STEP 1: INITIALIZATION AND ADDING NODES
# ==========================================
# Create the graph builder, passing our state structure
builder = StateGraph(AgentState)

# Register all nodes (node names are defined as strings)
builder.add_node("supervisor", supervisor_node)
builder.add_node("agent_for_analysing_tech_indicators", agent_for_analysing_tech_indicators)
# builder.add_node("agent_for_analysing_onchain_indicators", agent_for_analysing_onchain_indicators)
# builder.add_node("agent_for_news_analysis", agent_for_news_analysis)
builder.add_node("agent_for_twitter_analysis", agent_for_twitter_analysis)
# builder.add_node("agent_for_economic_calendar_analysis", agent_for_economic_calendar_analysis)
builder.add_node("validator", agent_for_verdicts_validation)
builder.add_node("agent_reports_analyser", agent_reports_analyser)

# ==========================================
# STEP 2: BUILDING EDGES (ROUTING)
# ==========================================
# 1. Entry point: from system start go to supervisor
builder.add_edge(START, "supervisor")

# 2. PARALLEL BRANCHING (Fan-out)
# Draw 4 edges from supervisor to agents.
# LangGraph will detect this and run them simultaneously!
builder.add_edge("supervisor", "agent_for_analysing_tech_indicators")
# builder.add_edge("supervisor", "agent_for_analysing_onchain_indicators")
# builder.add_edge("supervisor", "agent_for_news_analysis")
builder.add_edge("supervisor", "agent_for_twitter_analysis")
# builder.add_edge("supervisor", "agent_for_economic_calendar_analysis")

# 3. MERGE (Fan-in)
# The array means: "Wait for all these nodes to complete,
# and only then pass control to validator"
builder.add_edge(["agent_for_twitter_analysis", "agent_for_analysing_tech_indicators"], "validator")

# 4. Conditional exit: if there are agents with recompose_report=True — retry from supervisor
builder.add_conditional_edges("validator", _should_retry)

builder.add_edge("agent_reports_analyser", END)

# ==========================================
# STEP 3: GRAPH COMPILATION
# ==========================================
# At this stage LangGraph checks for dead ends in the graph
# and verifies that Reducers in State are configured correctly.
app = builder.compile()

# ==========================================
# STEP 4: RUN (INVOKE)
# ==========================================
if __name__ == "__main__":
    # Load multiagent system config
    config_path = Path(__file__).parent / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    
    N_days = 100

    load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")
    os.environ["COINGLASS_API_KEY"]  # fail fast if key is missing

    results_dataset = make_prediction_for_last_N_days(app, config, N_days)
    results_dataset = add_y_true(results_dataset, config["horizon"])

    output_path = Path(__file__).parent / "predictions_results.csv"
    results_dataset[["forecast_start_date", "y_predict", "y_predict_confidence", "y_true"]].to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved → {output_path}")

    cm_path = Path(__file__).parent / "confusion_matrix.png"
    build_confusion_matrix(results_dataset, config["horizon"], cm_path)