from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, AgentSignal, RetryAgentEntry
from typing import cast
from pydantic import BaseModel


class AgentValidationResult(BaseModel):
    has_problem: bool
    description: str  # Конкретное описание проблемы или "" если проблем нет


VALIDATOR_SYSTEM_PROMPT = """
Ты — валидатор аналитических отчётов агентов, прогнозирующих направление цены BTC.

Твоя задача: проверить отчёт агента (поля reasoning, summary и risks) только на ГРУБЫЕ, ОЧЕВИДНЫЕ ошибки.

Считай проблемой ТОЛЬКО:
1. Математические ошибки — явно неверные числа или прямое противоречие между числами в тексте (например, "RSI=90" при данных RSI=45).
2. Пустой или бессмысленный reasoning — раздел состоит из одного предложения или не содержит разбора ни одного индикатора.
3. Явное логическое противоречие — reasoning ОДНОЗНАЧНО описывает медвежий рынок по ВСЕМ показателям, но prediction=True (ВЫШЕ), или наоборот.
4. Критическое противоречие рисков — поле risks содержит ТОЛЬКО однозначно бычьи аргументы при prediction=False, или ТОЛЬКО медвежьи при prediction=True, и reasoning никак это не объясняет.

НЕ считай проблемой:
- Наличие смешанных сигналов (часть бычьих, часть медвежьих) при итоговом прогнозе в любую сторону — это нормальный анализ.
- Summary не упоминает все медвежьи сигналы при бычьем прогнозе — это допустимо.
- Осторожные или вероятностные формулировки ("возможно", "средняя уверенность").
- Субъективное несогласие с прогнозом.
- Пустое поле risks — это допустимо, если значимых рисков нет.

Отвечай строго через структуру: has_problem (bool) и description (строка с конкретным описанием ТОЛЬКО грубой ошибки или "" если проблем нет).

Prediction = False - значит что цена пойдет вниз
Prediction = True - значит что цена пойдет вверх
""".strip()


AGENT_DIR = Path(__file__).parent


def agent_for_verdicts_validation(state: AgentState):
    print(f"\n[validator] Начинаем валидацию {len(state['agent_signals'])} сигналов...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    updated_signals: dict[str, AgentSignal] = {}
    updated_retry: list[RetryAgentEntry] = [RetryAgentEntry(**entry) for entry in state["try_again_launch_agents"]]

    for agent_name, signal in state["agent_signals"].items():
        reasoning = signal.get("reasoning") or ""
        summary = signal.get("summary") or ""
        risks = signal.get("risks") or ""
        prediction = signal.get("prediction")
        prev_description = signal.get("description_of_the_reports_problem", "")

        # Пропускаем агентов-заглушки (нет ни reasoning, ни summary) — сбрасываем их флаг
        if not reasoning and not summary:
            print(f"[validator] {agent_name}: stub — пропускаем, флаг сброшен")
            updated_signals[agent_name] = signal
            for entry in updated_retry:
                if entry["agent_name"] == agent_name:
                    entry["recompose_report"] = False
                    break
            continue
        print(f"[validator] {agent_name}: проверяем отчёт...")

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Отчёт агента «{agent_name}»:\n\n"
                f"**reasoning:**\n{reasoning}\n\n"
                f"**summary:**\n{summary}\n\n"
                f"**risks:**\n{risks if risks else '(не указаны)'}\n\n"
                f"**prediction:** {'ВЫШЕ (True)' if prediction else 'НИЖЕ (False)'}"
            )),
        ]

        if prev_description:
            messages.append(HumanMessage(content=(
                f"Предыдущая проблема, зафиксированная валидатором:\n{prev_description}"
            )))

        result = cast(
            AgentValidationResult,
            llm.with_structured_output(AgentValidationResult).invoke(messages),
        )

        if result.has_problem:
            print(f"[validator] {agent_name}: ПРОБЛЕМА — {result.description[:1000]}")
        else:
            print(f"[validator] {agent_name}: OK")

        updated_signals[agent_name] = {
            **signal,
            "description_of_the_reports_problem": result.description if result.has_problem else "",
        }

        # Явно проставляем флаг — True если есть проблема, False если прошёл проверку
        for entry in updated_retry:
            if entry["agent_name"] == agent_name:
                entry["recompose_report"] = result.has_problem
                break

    return {
        "agent_signals": updated_signals,
        "try_again_launch_agents": updated_retry,
    }
