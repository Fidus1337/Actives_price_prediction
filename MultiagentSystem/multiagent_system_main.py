import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

from langgraph.graph import StateGraph, START, END
from multiagent_types import AgentState, _default_retry_agents
from agents.tech_indicators import agent_a_tech
from SharedDataCache.SharedBaseDataCache import SharedBaseDataCache
from agents.verdicts_validator import agent_for_verdicts_validation

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

def supervisor_node(state: AgentState):
    retry = state.get("retry_count", 0) + 1
    agents_to_run = [e["agent_name"] for e in state["try_again_launch_agents"] if e["recompose_report"]]
    if agents_to_run:
        print(f"\n[supervisor] Итерация #{retry} — повторный запуск агентов: {agents_to_run}")
    else:
        print(f"\n[supervisor] Итерация #{retry} — первый запуск всех агентов")
    return {"retry_count": retry}


MAX_RETRIES = 2

def _should_retry(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    agents_with_problems = [e["agent_name"] for e in state["try_again_launch_agents"] if e["recompose_report"]]

    if agents_with_problems and retry_count < MAX_RETRIES:
        print(f"\n[router] Проблемы у агентов: {agents_with_problems} — retry #{retry_count + 1}/{MAX_RETRIES}")
        return "supervisor"

    if agents_with_problems:
        print(f"\n[router] Лимит retry ({MAX_RETRIES}) исчерпан. Проблемные агенты: {agents_with_problems} — завершаем.")
    else:
        print(f"\n[router] Все агенты прошли валидацию — завершаем.")
    return END

def agent_b_onchain(state: AgentState):
    print("[agent_b] stub — пропускаем")
    return {"agent_signals": {"agent_b": {"summary": None}}}

def agent_c_news(state: AgentState):
    print("[agent_c] stub — пропускаем")
    return {"agent_signals": {"agent_c": {"summary": None}}}

def agent_d_twitter(state: AgentState):
    print("[agent_d] stub — пропускаем")
    return {"agent_signals": {"agent_d": {"summary": None}}}

# ==========================================
# ШАГ 1: ИНИЦИАЛИЗАЦИЯ И ДОБАВЛЕНИЕ УЗЛОВ
# ==========================================
# Создаем "строителя" графа, передавая ему нашу структуру состояния
builder = StateGraph(AgentState)

# Регистрируем все узлы (названия узлов задаем строками)
builder.add_node("supervisor", supervisor_node)
builder.add_node("agent_a", agent_a_tech)
builder.add_node("agent_b", agent_b_onchain)
builder.add_node("agent_c", agent_c_news)
builder.add_node("agent_d", agent_d_twitter)
builder.add_node("validator", agent_for_verdicts_validation)
# Агента E пока нет в коде, но его узел будет добавляться аналогично

# ==========================================
# ШАГ 2: ПОСТРОЕНИЕ РЕБЕР (МАРШРУТИЗАЦИЯ)
# ==========================================
# 1. Точка входа: от системного старта идем к супервизору
builder.add_edge(START, "supervisor")

# 2. ПАРАЛЛЕЛЬНОЕ РАЗВЕТВЛЕНИЕ (Fan-out)
# Проводим 4 стрелки от супервизора к агентам. 
# LangGraph увидит это и запустит их одновременно!
builder.add_edge("supervisor", "agent_a")
builder.add_edge("supervisor", "agent_b")
builder.add_edge("supervisor", "agent_c")
builder.add_edge("supervisor", "agent_d")

# 3. СЛИЯНИЕ (Fan-in)
# Массив в начале означает: "Дождись выполнения всех этих узлов, 
# и только потом передай управление в validator"
builder.add_edge(["agent_a", "agent_b", "agent_c", "agent_d"], "validator")

# 4. Условный выход: если есть агенты с recompose_report=True — повторяем с супервизора
builder.add_conditional_edges("validator", _should_retry)

# ==========================================
# ШАГ 3: КОМПИЛЯЦИЯ ГРАФА
# ==========================================
# На этом этапе LangGraph проверяет, нет ли тупиков в графе 
# и правильно ли настроены Reducer'ы в State.
app = builder.compile()

png = app.get_graph().draw_mermaid_png()
graph_path = Path(__file__).parent / "graph.png"
graph_path.write_bytes(png)
os.startfile(graph_path)

# ==========================================
# ШАГ 4: ЗАПУСК (INVOKE)
# ==========================================
if __name__ == "__main__":
    # Загружаем конфиг мультиагентной системы
    config_path = Path(__file__).parent / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Формируем стартовые данные, которые запросил пользователь.
    # Нам не нужно заполнять весь State целиком, только вводные данные!
    cache = SharedBaseDataCache(api_key=os.environ["COINGLASS_API_KEY"])
    forecast_date = datetime.strptime(config["forecast_start_date"], "%Y-%m-%d").date()

    initial_input = {
        "config": config,
        "cached_dataset": cache,
        "forecast_start_date": forecast_date,
        "try_again_launch_agents": _default_retry_agents(),
        "retry_count": 0,
    }
    
    print(forecast_date)

    print("🚀 Запуск мультиагентного графа...")
    
    # Метод invoke запускает выполнение и возвращает финальное состояние State
    final_state = app.invoke(initial_input)
    
    print("\n✅ Граф завершил работу! Собранные сигналы агентов:")
    
    # Красиво выводим то, что собрали параллельные агенты
    for agent_name, report in final_state.get("agent_signals", {}).items():
        prediction = report.get("prediction")
        prefix = f"[{'↑' if prediction is True else '↓' if prediction is False else '?'}] {agent_name.upper()}"
        print(f"\n{prefix}")
        if report['summary'] is not None:
            if "reasoning" in report:
                print(f"  reasoning : {report['reasoning'][:200]}...")
                print(f"  summary   : {report['summary'][:]}")
                print(f"  summary   : {report['risks'][:]}")