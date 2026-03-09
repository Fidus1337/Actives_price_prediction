from pathlib import Path
from typing import Optional, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState
from pydantic import BaseModel


class ReportsAnalysisResponse(BaseModel):
    reasoning: str
    summary: str
    risks: str
    prediction: Optional[bool]  # True = ВЫШЕ, False = НИЖЕ, None = неопределённо


_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
ANALYSER_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _prediction_to_direction(prediction: Optional[bool]) -> str | None:
    if prediction is True:
        return "LONG"
    if prediction is False:
        return "SHORT"
    return None


def agent_reports_analyser(state: AgentState):
    signals = state.get("agent_signals", {})

    # Собираем только реальные отчёты (не стабы)
    real_reports = {
        name: signal for name, signal in signals.items()
        if signal.get("summary") is not None and signal.get("reasoning")
    }

    if not real_reports:
        print("[reports_analyser] Нет реальных отчётов — пропускаем")
        return {
            "general_prediction_by_all_reports": None,
            "general_reports_summary": "",
            "general_reports_reasoning": "Нет отчётов агентов для анализа",
            "general_reports_risks": "",
        }

    print(f"[reports_analyser] Анализируем {len(real_reports)} отчёт(ов): {list(real_reports.keys())}")

    # Формируем текст отчётов для промпта
    reports_text_parts = []
    for name, signal in real_reports.items():
        pred = signal.get("prediction")
        pred_label = "ВЫШЕ (True)" if pred is True else "НИЖЕ (False)"
        confidence = signal.get("confidence", "unknown")
        reports_text_parts.append(
            f"### Агент: {name}\n"
            f"**reasoning:** {signal.get('reasoning', '')}\n\n"
            f"**summary:** {signal.get('summary', '')}\n\n"
            f"**risks:** {signal.get('risks', '(не указаны)')}\n\n"
            f"**prediction:** {pred_label}\n"
            f"**confidence:** {confidence}"
        )

    reports_text = "\n\n---\n\n".join(reports_text_parts)

    horizon = state.get("config", {}).get("horizon", "?")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    messages = [
        SystemMessage(content=ANALYSER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Горизонт прогноза: {horizon} дней\n\n"
            f"Отчёты агентов:\n\n{reports_text}"
        )),
    ]

    response = cast(
        ReportsAnalysisResponse,
        llm.with_structured_output(ReportsAnalysisResponse).invoke(messages),
    )

    direction = _prediction_to_direction(response.prediction)
    direction_label = direction or "NEUTRAL"
    print(f"[reports_analyser] Общий вердикт: {direction_label} | {response.summary[:120]}...")

    return {
        "general_prediction_by_all_reports": direction,
        "general_reports_summary": response.summary,
        "general_reports_reasoning": response.reasoning,
        "general_reports_risks": response.risks,
    }
