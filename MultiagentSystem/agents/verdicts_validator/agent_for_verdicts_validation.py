from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, AgentSignal, AgentRetry
from pydantic import BaseModel


class AgentValidationResult(BaseModel):
    has_problem: bool
    description: str        # Specific problem description or "" if no problems


_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
VALIDATOR_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def agent_for_verdicts_validation(state: AgentState):
    return {}
