import json
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from typing import cast
from pydantic import BaseModel

import os
from openai import AzureOpenAI

class TechAnalysisResponse(BaseModel):
    reasoning: str        # пошаговый разбор всех индикаторов по структуре из промпта
    summary: str          # краткое итоговое заключение: прогноз + уверенность + диапазон
    risks: str            # риски и контраргументы к прогнозу (2–5 пунктов или "")
    prediction: Optional[bool]  # True = ВЫШЕ, False = НИЖЕ, None = нейтральный (сигналы противоречивы)

AGENT_DIR = Path(__file__).parent


def agent_a_tech(state: AgentState):
    # We should try to find retry of the agent, by default, we have True value for retry
    retry_entry = next(
        (e for e in state["try_again_launch_agents"] if e["agent_name"] == "tech_analyser_agent"),
        None,
    )
    if retry_entry is None or not retry_entry["recompose_report"]:
        print("[agent_a_tech] recompose_report=False — пропускаем")
        return {}
    print("[agent_a_tech] Запускаем анализ технических индикаторов...")
    
    # 1. Достать настройки агента и общие параметры из конфига
    settings = get_agent_settings(state, "agent_for_analysing_tech_indicators")
    horizon = state["config"]["horizon"]

    forecast_date = state["forecast_start_date"]

    # 2. Взять DataFrame, отфильтровать нужные колонки и N дней до forecast_start_date
    df = state["cached_dataset"].get_base_df()
    cols = [c for c in settings["base_feats"] if c in df.columns]

    df = df.loc[df["date"] <= pd.Timestamp(forecast_date), ["date"] + cols].tail(settings["window_to_analysis"])
    print(df)

    last_date = df["date"].iloc[-1].date() if len(df) > 0 else None
    if last_date != forecast_date:
        print(f"[agent_tech] WARNING: data ends at {last_date}, expected {forecast_date}. "
              f"SharedBaseDataCache may have a data gap.")

    # 3. Конвертировать в JSON для промпта и сохранить для отладки
    data_json = df.to_json(orient="records", date_format="iso")
    (AGENT_DIR / "input_data.json").write_text(data_json, encoding="utf-8")

    # 4. Извлечь текущую цену закрытия (последняя строка)
    close_col = "spot_price_history__close"
    close_price = df[close_col].iloc[-1] if close_col in df.columns else "N/A"

    # 5. Загрузить системный промпт (из файла или inline)
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = settings["system_prompt"]

    # 6. Вызвать LLM с CoT: reasoning заполняется первым, summary — на его основе
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    from langchain_openai import AzureChatOpenAI



    prev_feedback: list[str] = (
        state.get("agent_signals", {})
        .get("tech_analyser_agent", {})
        .get("description_of_the_reports_problem", [])
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Данные за {settings['window_to_analysis']} дней до {forecast_date}:\n{data_json}\n\n"
            f"Текущая цена закрытия BTC: {close_price}\n"
            f"Горизонт прогноза: {horizon} дней (от {forecast_date})\n\n"
            f"Ответь по структуре:\n"
            f"1. ПРОГНОЗ: будет ли цена выше, ниже или нейтрально через {horizon} дней?\n"
            f"   - True  = цена ВЫШЕ {close_price} через {horizon} дней\n"
            f"   - False = цена НИЖЕ {close_price} через {horizon} дней\n"
            f"   - null  = сигналы противоречивы, уверенность низкая — воздержаться от прогноза\n"
            f"2. АРГУМЕНТЫ: какие индикаторы поддерживают твой прогноз?\n"
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

    response = cast(TechAnalysisResponse, llm.with_structured_output(TechAnalysisResponse).invoke(messages))

    prediction_label = "ВЫШЕ" if response.prediction else "НИЖЕ"
    print(f"[agent_a_tech] Готово. Прогноз: {prediction_label} | summary: {response.summary[:120]}...")

    tech_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "base_feats": cols,
        "window": settings["window_to_analysis"],
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
    }
    (AGENT_DIR / "tech_predict.json").write_text(
        json.dumps(tech_predict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[agent_a_tech] tech_predict.json сохранён в {AGENT_DIR}")

    return {"agent_signals": {"tech_analyser_agent": {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
    }}}
