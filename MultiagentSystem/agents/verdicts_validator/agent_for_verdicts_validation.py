from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from typing import cast
from pydantic import BaseModel

AGENT_DIR = Path(__file__).parent

def agent_for_verdicts_validation(state: AgentState):
    # 6. Вызвать LLM с CoT: reasoning заполняется первым, summary — на его основе
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    for agents_prediction in state.agent_signals:
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
