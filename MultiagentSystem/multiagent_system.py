from datetime import date
from typing import List, Literal, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from SharedDataCache import SharedBaseDataCache
from dotenv import load_dotenv
import os

# Getting variable for caching database
#================================================================
load_dotenv("dev.env")

_api_key = os.getenv("COINGLASS_API_KEY")
if not _api_key:
    raise ValueError("COINGLASS_API_KEY not found in dev.env")
API_KEY: str = _api_key

__cached_database = SharedBaseDataCache(api_key=API_KEY)
#================================================================

class AgentSignal(TypedDict):
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float
    weight: float
    summary: Optional[str]


class RetryAgentEntry(TypedDict):
    agent_name: str
    recompose_report: bool


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    forecast_start_date: date
    close_price_by_start_date: float
    direction: Literal["LONG", "SHORT"]
    confidence: float
    horizon: int
    reasoning: str
    agent_signals: dict[str, AgentSignal]
    error_detected: bool
    error_reasoning: str
    try_again_launch_agents: list[RetryAgentEntry]