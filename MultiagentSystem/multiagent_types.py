from datetime import date
from typing import List, Literal, Union, Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache

# Reducer: безопасно сливает словари агентов
def merge_dicts(left: dict, right: dict) -> dict:
    if not left: left = {}
    if not right: right = {}
    return {**left, **right}

class AgentSignal(TypedDict):
    summary: str

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
    confidence: float
    reasoning: str
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts]
    error_detected: bool
    error_reasoning: str
    try_again_launch_agents: list[RetryAgentEntry]


def get_agent_settings(state: AgentState, agent_name: str) -> dict:
    """Получить настройки конкретного агента из state."""
    return state["config"]["agent_settings"][agent_name]
