import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, get_agent_settings
from dotenv import load_dotenv

load_dotenv("dev.env")


def agent_a_tech(state: AgentState):
    # 1. Достать настройки агента и общие параметры из конфига
    settings = get_agent_settings(state, "agent_for_analysing_tech_indicators")
    horizon = state["config"]["horizon"]

    # 2. Взять DataFrame, отфильтровать нужные колонки и последние N дней
    df = state["cached_dataset"].get_base_df()
    cols = [c for c in settings["base_feats"] if c in df.columns]
    df = df[["date"] + cols].tail(settings["window_to_analysis"])

    # 3. Конвертировать в JSON для промпта
    data_json = df.to_json(orient="records", date_format="iso")

    # 4. Вызвать LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    response = llm.invoke([
        SystemMessage(content=settings["system_prompt"]),
        HumanMessage(content=(
            f"Данные за последние {settings['window_to_analysis']} дней:\n{data_json}\n\n"
            f"Дай прогноз на {horizon} дней. "
            "Ответь строго в JSON: {\"signal\": \"BULLISH|BEARISH|NEUTRAL\", \"confidence\": 0.0-1.0, \"summary\": \"...\"}"
        ))
    ])

    # 5. Распарсить JSON из ответа (LLM может обернуть в ```json ... ```)
    text = response.content.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    result = json.loads(match.group() if match else text)

    return {"agent_signals": {"technical": {
        "signal": result["signal"],
        "confidence": result["confidence"],
        "weight": 0.2,
        "summary": result["summary"]
    }}}
