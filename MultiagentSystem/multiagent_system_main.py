import json
import os
import sys
from datetime import datetime
from pathlib import Path
from .multiagent_predictions_module import make_prediction_for_last_N_days
from .multiagent_predictions_module import build_confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

# For START END nodes
from langgraph.graph import StateGraph, START, END

# General Agent state for all agents
from multiagent_types import AgentState

# Tech agent
from .agents.tech_indicators import agent_for_analysing_tech_indicators
# On-chain agent
from .agents.onchain_indicators import agent_for_analysing_onchain_indicators
# Agent for analysing news related to crypto (Coinglass endpoint)
from .agents.news_analyser.agent_for_news_analysis import agent_for_news_analysis
# Agent for analysing macro-economic calendar events
from .agents.economic_calendar_analyser.agent_for_economic_calendar_analysis import agent_for_economic_calendar_analysis
# Agent for analysing Twitter sentiment signals
from .agents.twitter_analyser.agent_for_twitter_analysis import agent_for_twitter_analysis

# This agent checks if the report of the agent is logic and structured by all requirements
from .agents.verdicts_validator import agent_for_verdicts_validation
# This agent looks at all agents reports and makes final decision (skip predict / price will be higher / lower)
from .agents.reports_analyser import agent_reports_analyser

# The class for fetching dataset (1000 last days)
from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache
# Making Y column by horizon from config
from FeaturesEngineer.FeaturesEngineer import FeaturesEngineer

# Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Node for starting analysis by agents
def supervisor_node(state: AgentState):
    retry_agents = state.get("retry_agents", [])
    retry_counts = state.get("retry_counts", {})

    if retry_agents:
        # Increment counters only for agents that need retry
        updated_counts = {a: retry_counts.get(a, 0) + 1 for a in retry_agents}
        print(f"\n[supervisor] Restarting agents: {retry_agents}, counts: {updated_counts}")
        return {"retry_counts": updated_counts}
    else:
        print(f"\n[supervisor] First run of all agents")
        return {}

# Max retries per agent (default 2)
MAX_RETRIES = 2
# Router for retry (look schema at miro)
def _should_retry(state: AgentState) -> str:
    retry_agents = state.get("retry_agents", [])
    retry_counts = state.get("retry_counts", {})

    # Filter to agents that still have retries left
    agents_with_budget = [
        a for a in retry_agents
        if retry_counts.get(a, 0) < MAX_RETRIES
    ]

    if agents_with_budget:
        print(f"\n[router] Agents need retry: {agents_with_budget} — counts: {retry_counts}")
        return "supervisor"

    if retry_agents:
        exhausted = [a for a in retry_agents if retry_counts.get(a, 0) >= MAX_RETRIES]
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
builder.add_node("agent_for_analysing_onchain_indicators", agent_for_analysing_onchain_indicators)
builder.add_node("agent_for_news_analysis", agent_for_news_analysis)
builder.add_node("agent_for_twitter_analysis", agent_for_twitter_analysis)
builder.add_node("agent_for_economic_calendar_analysis", agent_for_economic_calendar_analysis)
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
builder.add_edge("supervisor", "agent_for_analysing_onchain_indicators")
builder.add_edge("supervisor", "agent_for_news_analysis")
builder.add_edge("supervisor", "agent_for_twitter_analysis")
builder.add_edge("supervisor", "agent_for_economic_calendar_analysis")

# 3. MERGE (Fan-in)
# The array means: "Wait for all these nodes to complete,
# and only then pass control to validator"
builder.add_edge(["agent_for_twitter_analysis", "agent_for_analysing_tech_indicators", "agent_for_economic_calendar_analysis", "agent_for_news_analysis", "agent_for_analysing_onchain_indicators"], "validator")

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
    
    horizon = int(config["horizon"])
    N = 10

    results_dataset = make_prediction_for_last_N_days(config, N)

    # Final confusion matrix
    cm_path = Path(__file__).parent / "agents" / "tech_agent_confusion_matrix.png"
    done_count = 0
    build_confusion_matrix(results_dataset, N, horizon, cm_path)

    print("\n✅ Graph finished! Collected agent signals:")

    # Save results of predictions to csv
    output_path = Path(__file__).parent / "tech_agent_results.csv"
    results_dataset.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")