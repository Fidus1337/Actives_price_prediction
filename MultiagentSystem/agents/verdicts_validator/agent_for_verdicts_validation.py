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

# Agents that are NOT validated (formula-based, no LLM report to check)
_SKIP_AGENTS = frozenset({"agent_for_twitter_analysis"})

TAG = "[validator]"


def agent_for_verdicts_validation(state: AgentState):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Build mutable lookup: agent_name → copy of AgentRetry entry
    retry_map: dict[str, AgentRetry] = {r["agent_name"]: cast(AgentRetry, dict(r)) for r in state.get("retry_agents", [])}

    updated_retry: list[AgentRetry] = []
    updated_signals: dict[str, AgentSignal] = {}

    for agent_name, signal in (state.get("agent_signals") or {}).items():
        # Skip formula-based agents (twitter etc.) — nothing to validate
        if agent_name in _SKIP_AGENTS:
            print(f"{TAG} {agent_name}: skipped (formula-based agent)")
            continue

        # Safety net: skip anything without a retry entry
        if agent_name not in retry_map:
            print(f"{TAG} {agent_name}: skipped (no retry entry)")
            continue

        # Skip agents that returned no reasoning (e.g. skipped on retry)
        if not signal.get("reasoning"):
            print(f"{TAG} {agent_name}: skipped (no reasoning in signal)")
            continue

        prediction = signal.get("prediction")
        prediction_label = "True (HIGHER)" if prediction else "False (LOWER)"

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Agent: {agent_name}\n"
                f"Reasoning: {signal.get('reasoning', '')}\n"
                f"Summary: {signal.get('summary', '')}\n"
                f"Risks: {signal.get('risks', '')}\n"
                f"Prediction: {prediction_label}\n"
            )),
        ]

        try:
            result = cast(
                AgentValidationResult,
                llm.with_structured_output(AgentValidationResult).invoke(messages),
            )
        except Exception as exc:
            print(f"{TAG} {agent_name}: LLM call failed — {exc}, skipping validation")
            continue

        if result.has_problem:
            entry = retry_map[agent_name]
            entry["retry_requirements"] = list(entry["retry_requirements"]) + [result.description]
            entry["currents_retry"] += 1
            updated_retry.append(entry)

            updated_signal = cast(AgentSignal, dict(signal))
            problems = list(signal.get("description_of_the_reports_problem") or [])
            problems.append(result.description)
            updated_signal["description_of_the_reports_problem"] = problems
            updated_signals[agent_name] = updated_signal

            print(f"{TAG} {agent_name}: PROBLEM (attempt {entry['currents_retry']}/{entry['max_retries']}) — {result.description}")
        else:
            print(f"{TAG} {agent_name}: OK")

    return {"retry_agents": updated_retry, "agent_signals": updated_signals}
