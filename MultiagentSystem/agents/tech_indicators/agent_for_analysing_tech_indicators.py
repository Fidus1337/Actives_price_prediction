from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from typing import cast
from pydantic import BaseModel


class TechAnalysisResponse(BaseModel):
    reasoning: str   # пошаговый разбор всех индикаторов по структуре из промпта
    summary: str     # краткое итоговое заключение: прогноз + уверенность + диапазон
    prediction: bool # True = цена будет ВЫШЕ, False = НИЖЕ — вывод из reasoning и summary

AGENT_DIR = Path(__file__).parent


def agent_a_tech(state: AgentState):
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

    response = cast(TechAnalysisResponse, llm.with_structured_output(TechAnalysisResponse).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Данные за {settings['window_to_analysis']} дней до {forecast_date}:\n{data_json}\n\n"
            f"Текущая цена закрытия BTC: {close_price}\n"
            f"Горизонт прогноза: {horizon} дней (от {forecast_date})\n\n"
            f"Ответь по структуре:\n"
            f"1. ПРОГНОЗ: будет ли цена выше или ниже {close_price} через {horizon} дней?\n"
            f"2. АРГУМЕНТЫ: какие индикаторы поддерживают твой прогноз?\n"
        ))
    ]))

    return {"agent_signals": {"technical": {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "prediction": response.prediction,
    }}}
