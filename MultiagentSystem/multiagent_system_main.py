import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

# For START END nodes
from langgraph.graph import StateGraph, START, END

# General Agent state for all agents
from multiagent_types import AgentState

# Tech agent
from agents.tech_indicators import agent_a_tech
# On-chain agent
from agents.onchain_indicators import agent_b_onchain
# Agent for analysing news related to crypto (Coinglass endpoint)
from agents.news_analyser import agent_c_news

# This agent checks if the report of the agent is logic and structured by all requirements
from agents.verdicts_validator import agent_for_verdicts_validation
# This agent looks at all agents reports and makes final decision (skip predict / price will be higher / lower)
from agents.reports_analyser import agent_reports_analyser

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

# FUTURE LOGIC FOR ANALYSIS TWITTER
def agent_d_twitter(state: AgentState):
    print("[agent_d] stub — skipping")
    return {"agent_signals": {"agent_d": {"summary": None}}}

# ==========================================
# STEP 1: INITIALIZATION AND ADDING NODES
# ==========================================
# Create the graph builder, passing our state structure
builder = StateGraph(AgentState)

# Register all nodes (node names are defined as strings)
builder.add_node("supervisor", supervisor_node)
builder.add_node("agent_a_tech", agent_a_tech)
builder.add_node("agent_b_onchain", agent_b_onchain)
builder.add_node("agent_c_news", agent_c_news)
builder.add_node("agent_d_twitter", agent_d_twitter)
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
builder.add_edge("supervisor", "agent_a_tech")
# builder.add_edge("supervisor", "agent_b_onchain")
builder.add_edge("supervisor", "agent_c_news")

# 3. MERGE (Fan-in)
# The array means: "Wait for all these nodes to complete,
# and only then pass control to validator"
builder.add_edge(["agent_c_news", "agent_a_tech"], "validator")

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
    # Building matrix by N last days
    N_last_dates = 25
    
    # Load multiagent system config
    config_path = Path(__file__).parent / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Build dataset without last {horizon} samples
    cache = SharedBaseDataCache(api_key=os.environ["COINGLASS_API_KEY"])
    forecast_date = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d").date()

    horizon = config["horizon"]
    base_df = cache.get_base_df()
    dataset_with_target = FeaturesEngineer().add_y_up_custom(
        base_df, horizon=horizon, close_col="spot_price_history__close"
    )
    dataset_with_target = dataset_with_target.head(-horizon)
    
    print("Loaded dataset, look at the last date allowed for building counfusion matrix: \n")
    print(dataset_with_target[["date", "spot_price_history__close", f"y_up_{horizon}d"]].tail(1))
    
    # Use forecast_start_date from config as the anchor: take N_last_dates ending at that date
    forecast_anchor = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d")
    eligible = dataset_with_target[dataset_with_target["date"] <= forecast_anchor]
    results_dataset = eligible[["date", f"y_up_{horizon}d"]].tail(N_last_dates).copy()
    print(f"\nUsing forecast_start_date={config['forecast_start_date']} as anchor")
    print(f"Eligible rows (date <= anchor): {len(eligible)} | Taking last {N_last_dates}")
    results_dataset["y_predictions"] = None
    results_dataset["confidence_score"] = None

    # Where we store confusion matrix
    cm_path = Path(__file__).parent / "agents" / "tech_agent_confusion_matrix.png"
    done_count = 0
    print(f"\nStarting predictions loop: {len(results_dataset)} dates, horizon={horizon}d")
    print(f"Date range: {results_dataset['date'].iloc[0].date()} -> {results_dataset['date'].iloc[-1].date()}")

    total_rows = len(results_dataset)
    for idx, row in results_dataset.iterrows():              # iterate over indices
        forecast_date = row["date"].date()                   # Python date, not datetime64
        done_count += 1

        print(f"\n{'#'*70}")
        print(f"### PREDICTION {done_count}/{total_rows} | date={forecast_date} | y_true={row[f'y_up_{horizon}d']}")
        print(f"{'#'*70}")

        initial_input = {
            "config": config,
            "cached_dataset": base_df,
            "forecast_start_date": forecast_date,
            "retry_agents": [],
            "retry_counts": {},
        }

        final_state = app.invoke(initial_input)

        # Get the overall verdict from reports_analyser
        direction = final_state.get("general_prediction_by_all_reports")
        score = final_state.get("confidence_score", 0)
        pred = 1 if direction == "LONG" else (0 if direction == "SHORT" else None)
        results_dataset.at[idx, "y_predictions"] = pred
        results_dataset.at[idx, "confidence_score"] = score

        actual = int(row[f"y_up_{horizon}d"])
        correct = "CORRECT" if pred == actual else ("WRONG" if pred is not None else "SKIPPED")
        print(f"\n>>> RESULT: prediction={direction or 'NEUTRAL'} (score={score}) | y_true={actual} | {correct}")

        # Update confusion matrix every 10 predictions
        if done_count % 10 == 0:
            valid = results_dataset.dropna(subset=["y_predictions"])
            if len(valid) >= 2:
                y_true = valid[f"y_up_{horizon}d"].astype(int)
                y_pred = valid["y_predictions"].astype(int)
                cm = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(cm, display_labels=["LOWER (0)", "HIGHER (1)"])
                disp.plot()
                plt.title(f"Confusion Matrix ({done_count}/{N_last_dates} predictions, horizon={horizon}d)")
                plt.savefig(cm_path)
                plt.close()
                print(f"[CM] Updated confusion matrix ({done_count} predictions) → {cm_path}")

    print(results_dataset[["date", f"y_up_{horizon}d", "y_predictions", "confidence_score"]])

    output_path = Path(__file__).parent / "tech_agent_results.csv"
    results_dataset.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Final confusion matrix
    valid = results_dataset.dropna(subset=["y_predictions"])
    if len(valid) >= 2:
        y_true = valid[f"y_up_{horizon}d"].astype(int)
        y_pred = valid["y_predictions"].astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["LOWER (0)", "HIGHER (1)"])
        disp.plot()
        plt.title(f"Final Confusion Matrix ({len(valid)}/{N_last_dates} predictions, horizon={horizon}d)")
        plt.savefig(cm_path)
        plt.close()
        print(f"Final confusion matrix saved to {cm_path}")

    print("\n✅ Graph finished! Collected agent signals:")