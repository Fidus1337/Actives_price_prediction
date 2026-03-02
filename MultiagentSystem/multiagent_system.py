from datetime import date
from typing import List, Literal, Optional, Union, Annotated # <-- Добавили Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
import operator # <-- Понадобится для Reducer
from langgraph.graph import StateGraph, START, END

# ... ваши импорты и кэш ...

class AgentSignal(TypedDict):
    signal: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float
    weight: float
    summary: Optional[str]

class RetryAgentEntry(TypedDict):
    agent_name: str
    recompose_report: bool

# Пишем простую функцию-reducer, которая будет безопасно сливать словари агентов
def merge_dicts(left: dict, right: dict) -> dict:
    if not left: left = {}
    if not right: right = {}
    return {**left, **right} # Объединяем старый словарь с новым

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    forecast_start_date: date
    close_price_by_start_date: float
    direction: Literal["LONG", "SHORT"]
    confidence: float
    horizon: int
    reasoning: str
    
    # ВОТ ОНО МАГИЧЕСКОЕ ИСПРАВЛЕНИЕ: 
    # Теперь LangGraph знает, что данные от 4-х агентов нужно "сливать" через merge_dicts
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts]
    
    error_detected: bool
    error_reasoning: str
    try_again_launch_agents: list[RetryAgentEntry]

def supervisor_node(state: AgentState):
    # Ничего не меняем, просто пропускаем дальше
    return {} # <-- Возвращаем пустой словарь, а не state!

def agent_a_tech(state: AgentState):
    return {"agent_signals": {"technical": {"signal": "BULLISH", "confidence": 0.8, "weight": 0.2, "summary": None}}}

def agent_b_onchain(state: AgentState):
    return {"agent_signals": {"onchain": {"signal": "BEARISH", "confidence": 0.6, "weight": 0.3, "summary": None}}}

def agent_c_news(state: AgentState):
    return {"agent_signals": {"news_background": {"signal": "NEUTRAL", "confidence": 0.5, "weight": 0.2, "summary": None}}}

def agent_d_twitter(state: AgentState):
    return {"agent_signals": {"twitter_news": {"signal": "BULLISH", "confidence": 0.7, "weight": 0.3, "summary": None}}}

def validation_node(state: AgentState):
    print("Собранные сигналы:", state.get("agent_signals"))
    return {} # <-- Тоже возвращаем пустой словарь

from langgraph.graph import StateGraph, START, END
from datetime import date

# Предполагается, что AgentState и функции узлов (agent_a_tech, и т.д.) 
# уже определены выше в коде.

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
builder.add_node("validator", validation_node)
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

# 4. Точка выхода: завершаем работу после валидатора
builder.add_edge("validator", END)

# ==========================================
# ШАГ 3: КОМПИЛЯЦИЯ ГРАФА
# ==========================================
# На этом этапе LangGraph проверяет, нет ли тупиков в графе 
# и правильно ли настроены Reducer'ы в State.
app = builder.compile()

# ==========================================
# ШАГ 4: ЗАПУСК (INVOKE)
# ==========================================
if __name__ == "__main__":
    # Формируем стартовые данные, которые запросил пользователь.
    # Нам не нужно заполнять весь State целиком, только вводные данные!
    initial_input = {
        "forecast_start_date": date(2026, 3, 2),
        "close_price_by_start_date": 65000.0,
        "horizon": 7
    }

    print("🚀 Запуск мультиагентного графа...")
    
    # Метод invoke запускает выполнение и возвращает финальное состояние State
    final_state = app.invoke(initial_input)
    
    print("\n✅ Граф завершил работу! Собранные сигналы агентов:")
    
    # Красиво выводим то, что собрали параллельные агенты
    for agent_name, report in final_state.get("agent_signals", {}).items():
        print(f" - {agent_name.upper()}: Сигнал {report['signal']}, Уверенность: {report['confidence']}")