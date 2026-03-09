from datetime import date
from typing import List, Literal, Union, Annotated

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict


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
    prediction: bool  # True = ВЫШЕ, False = НИЖЕ (агенты всегда выбирают направление)
    confidence: str   # high / medium / low

class AgentState(TypedDict):
    config: dict
    cached_dataset: pd.DataFrame
    general_prediction_by_all_reports: Literal["LONG", "SHORT"] | None
    general_reports_summary: str
    general_reports_reasoning: str
    general_reports_risks: str
    horizon: int
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    forecast_start_date: date
    reasoning: str
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts]
    error_detected: bool
    error_reasoning: str
    retry_agents: list[str]   # имена агентов, которым нужен retry (пустой = первый запуск / всё ОК)
    retry_count: int

def get_agent_settings(state: AgentState, agent_name: str) -> dict:
    """Получить настройки конкретного агента из state."""
    return state["config"]["agent_settings"][agent_name]
