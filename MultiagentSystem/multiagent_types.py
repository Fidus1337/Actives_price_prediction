from datetime import date
from typing import List, Literal, Union, Annotated

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache

# Reducer: безопасно сливает словари агентов
def merge_dicts(left: dict, right: dict) -> dict:
    if not left: left = {}
    if not right: right = {}
    return {**left, **right}

class AgentSignal(TypedDict):
    description_of_the_reports_problem: list[str]
    reasoning: str
    summary: str
    risks: str        # риски и контраргументы к прогнозу
    prediction: bool | None  # True = ВЫШЕ, False = НИЖЕ, None = нейтральный/неопределённый

class RetryAgentEntry(TypedDict):
    agent_name: str
    recompose_report: bool

class AgentState(TypedDict):
    config: dict
    horizon: int
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    cached_dataset: SharedBaseDataCache
    forecast_start_date: date
    direction: Literal["LONG", "SHORT"]
    reasoning: str
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts]
    error_detected: bool
    error_reasoning: str
    try_again_launch_agents: list[RetryAgentEntry]
    retry_count: int

def _default_retry_agents() -> list[RetryAgentEntry]:
    return [
        {"agent_name": "tech_analyser_agent", "recompose_report": True},
        {"agent_name": "agent_b",             "recompose_report": True},
        {"agent_name": "agent_c",             "recompose_report": True},
        {"agent_name": "agent_d",             "recompose_report": True},
    ]


def get_agent_settings(state: AgentState, agent_name: str) -> dict:
    """Получить настройки конкретного агента из state."""
    return state["config"]["agent_settings"][agent_name]
