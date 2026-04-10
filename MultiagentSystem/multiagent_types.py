from datetime import date
from typing import List, Literal, Union, Annotated

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict


# Reducer для agent_signals — мержит словари по ключу
def merge_dicts(left: dict, right: dict) -> dict:
    if not left: left = {}
    if not right: right = {}
    return {**left, **right}


# Reducer для retry_agents — заменяет AgentRetry по agent_name, добавляет новые
def merge_retry_agents(left: list, right: list) -> list:
    if not left: left = []
    if not right: right = []
    # Берём существующий список и заменяем/добавляем элементы из right по agent_name
    result = {r["agent_name"]: r for r in left}
    for r in right:
        result[r["agent_name"]] = r  # перезаписывает старый если agent_name совпадает
    return list(result.values())

# This is class, which exists for structuring reports by agents
class AgentSignal(TypedDict):
    description_of_the_reports_problem: list[str]
    reasoning: str
    summary: str
    risks: str        # contrarguments about end prediction
    prediction: bool  # True(up)/False(down)
    confidence: str   # high / medium / low

class AgentRetry(TypedDict):
    agent_name: str
    max_retries: int
    currents_retry: int
    retry_requirements: list[str]

# General agent state
class AgentState(TypedDict):
    config: dict # config from MultiagentSystem folder, all settings for launching precitions
    agent_envolved_in_prediction: list[str]
    cached_dataset: pd.DataFrame
    horizon: int
    general_prediction_by_all_reports: Literal["LONG", "SHORT"] | None # after analysis by agent_reports_analyser, we can skip predicts
    general_reports_summary: str
    general_reports_reasoning: str
    general_reports_risks: str
    confidence_score: int
    forecast_start_date: str 
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts] # every agent returns signal
    retry_agents: Annotated[list[AgentRetry], merge_retry_agents]

def get_agent_settings(state: AgentState, agent_name: str) -> dict:
    """Get settings for a specific agent from state."""
    return state["config"]["agent_settings"][agent_name]
