import json
from pathlib import Path
from typing import cast

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel


class OnchainAnalysisResponse(BaseModel):
    reasoning: str
    summary: str
    risks: str
    prediction: bool       # True = ВЫШЕ, False = НИЖЕ (всегда выбирай направление)
    confidence: str        # high / medium / low


AGENT_DIR = Path(__file__).parent
AGENT_NAME = "onchain_analyser_agent"
CONFIG_KEY = "agent_for_analysing_onchain_indicators"


def agent_b_onchain(state: AgentState):
    retry_agents = state.get("retry_agents", [])
    is_first_run = state.get("retry_count", 0) <= 1
    if not is_first_run and AGENT_NAME not in retry_agents:
        print(f"[agent_b_onchain] retry не требуется — пропускаем")
        return {}

    print("[agent_b_onchain] Запускаем анализ on-chain индикаторов...")

    settings = get_agent_settings(state, CONFIG_KEY)
    horizon = state["config"]["horizon"]
    forecast_date = state["forecast_start_date"]

    # Взять DataFrame, отфильтровать нужные колонки и окно
    df = state["cached_dataset"].copy()
    cols = [c for c in settings["base_feats"] if c in df.columns]

    df = df.loc[df["date"] <= pd.Timestamp(forecast_date), ["date"] + cols].tail(settings["window_to_analysis"])

    last_date = df["date"].iloc[-1].date() if len(df) > 0 else None
    if last_date != forecast_date:
        print(f"[agent_b_onchain] WARNING: data ends at {last_date}, expected {forecast_date}.")

    # Конвертировать в JSON для промпта
    data_json = df.to_json(orient="records", date_format="iso")
    (AGENT_DIR / "input_data.json").write_text(data_json, encoding="utf-8")

    # Цена закрытия
    close_col = "spot_price_history__close"
    close_price = df[close_col].iloc[-1] if close_col in df.columns else "N/A"

    # Системный промпт
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = settings["system_prompt"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Обратная связь от валидатора
    prev_feedback: list[str] = (
        state.get("agent_signals", {})
        .get(AGENT_NAME, {})
        .get("description_of_the_reports_problem", [])
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"On-chain данные за {settings['window_to_analysis']} дней до {forecast_date}:\n{data_json}\n\n"
            f"Текущая цена закрытия BTC: {close_price}\n"
            f"Горизонт прогноза: {horizon} дней (от {forecast_date})\n\n"
            f"Ответь по структуре:\n"
            f"1. ПРОГНОЗ: будет ли цена выше или ниже через {horizon} дней?\n"
            f"   - True  = цена ВЫШЕ {close_price} через {horizon} дней\n"
            f"   - False = цена НИЖЕ {close_price} через {horizon} дней\n"
            f"   ВСЕГДА выбирай направление (True или False). Укажи уровень уверенности в поле confidence.\n"
            f"2. АРГУМЕНТЫ: какие on-chain метрики поддерживают твой прогноз?\n"
        )),
    ]

    if prev_feedback:
        history_text = "\n".join(
            f"Итерация {i+1}: {d}" for i, d in enumerate(prev_feedback)
        )
        messages.append(HumanMessage(content=(
            f"ЗАМЕЧАНИЯ ВАЛИДАТОРА ПО ПРЕДЫДУЩИМ ВЕРСИЯМ ОТЧЁТА:\n{history_text}\n\n"
            f"Учти эти замечания при составлении нового отчёта."
        )))

    response = cast(OnchainAnalysisResponse, llm.with_structured_output(OnchainAnalysisResponse).invoke(messages))

    pred_label = "ВЫШЕ" if response.prediction else "НИЖЕ"
    print(f"[agent_b_onchain] Готово. Прогноз: {pred_label} | summary: {response.summary[:120]}...")

    onchain_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "base_feats": cols,
        "window": settings["window_to_analysis"],
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }
    (AGENT_DIR / "onchain_predict.json").write_text(
        json.dumps(onchain_predict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[agent_b_onchain] onchain_predict.json сохранён в {AGENT_DIR}")

    return {"agent_signals": {AGENT_NAME: {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
    }}}
