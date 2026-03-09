from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, AgentSignal, RetryAgentEntry
from pydantic import BaseModel


class AgentValidationResult(BaseModel):
    has_problem: bool
    description: str        # Конкретное описание проблемы или "" если проблем нет
    should_nullify: bool    # True = уверенность низкая, обнулить prediction → None


_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
VALIDATOR_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _set_retry_flag(retry_list: list[RetryAgentEntry], agent_name: str, value: bool) -> None:
    for entry in retry_list:
        if entry["agent_name"] == agent_name:
            entry["recompose_report"] = value
            break


def agent_for_verdicts_validation(state: AgentState):
    print(f"\n[validator] Начинаем валидацию {len(state['agent_signals'])} сигналов...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    updated_signals: dict[str, AgentSignal] = {}
    updated_retry: list[RetryAgentEntry] = [RetryAgentEntry(**e) for e in state["try_again_launch_agents"]]

    for agent_name, signal in state["agent_signals"].items():
        reasoning = signal.get("reasoning") or ""
        summary   = signal.get("summary") or ""
        risks     = signal.get("risks") or ""
        prediction = signal.get("prediction")
        raw = signal.get("description_of_the_reports_problem", [])
        prev_descriptions = [raw] if isinstance(raw, str) and raw else (raw or [])

        if not reasoning and not summary:
            print(f"[validator] {agent_name}: stub — пропускаем, флаг сброшен")
            updated_signals[agent_name] = signal
            _set_retry_flag(updated_retry, agent_name, False)
            continue

        print(f"[validator] {agent_name}: проверяем отчёт...")

        pred_label = "ВЫШЕ (True)" if prediction is True else ("НИЖЕ (False)" if prediction is False else "НЕЙТРАЛЬНО (None)")
        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Отчёт агента «{agent_name}»:\n\n"
                f"**reasoning:**\n{reasoning}\n\n"
                f"**summary:**\n{summary}\n\n"
                f"**risks:**\n{risks or '(не указаны)'}\n\n"
                f"**prediction:** {pred_label}"
            )),
        ]

        if prev_descriptions:
            history = "\n".join(f"Итерация {i+1}: {d}" for i, d in enumerate(prev_descriptions))
            messages.append(HumanMessage(content=f"История предыдущих замечаний валидатора:\n{history}"))

        result = cast(AgentValidationResult, llm.with_structured_output(AgentValidationResult).invoke(messages))

        print(f"[validator] {agent_name}: {'ПРОБЛЕМА — ' + result.description[:1000] if result.has_problem else 'OK'}")

        if result.should_nullify:
            print(f"[validator] {agent_name}: низкая уверенность — prediction обнулён до None")
            final_prediction = None
        else:
            final_prediction = prediction

        updated_signals[agent_name] = {
            **signal,
            "prediction": final_prediction,
            "description_of_the_reports_problem": prev_descriptions + ([result.description] if result.has_problem else []),
        }
        _set_retry_flag(updated_retry, agent_name, result.has_problem)

    return {"agent_signals": updated_signals, "try_again_launch_agents": updated_retry}
