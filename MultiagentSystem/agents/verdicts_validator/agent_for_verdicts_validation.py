from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from multiagent_types import AgentState, AgentSignal
from pydantic import BaseModel


class AgentValidationResult(BaseModel):
    has_problem: bool
    description: str        # Specific problem description or "" if no problems


_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
VALIDATOR_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def agent_for_verdicts_validation(state: AgentState):
    TAG = "[validator]"
    signals = state["agent_signals"]

    print(f"\n{'='*60}")
    print(f"{TAG} === VALIDATING {len(signals)} AGENT REPORTS ===")
    print(f"{'='*60}")
    print(f"{TAG} Agents to validate: {list(signals.keys())}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    updated_signals: dict[str, AgentSignal] = {}
    retry_agents: list[str] = []

    for i, (agent_name, signal) in enumerate(signals.items(), 1):
        print(f"\n{TAG} --- Checking agent {i}/{len(signals)}: {agent_name} ---")

        reasoning = signal.get("reasoning") or ""
        summary   = signal.get("summary") or ""
        risks     = signal.get("risks") or ""
        prediction = signal.get("prediction")
        raw = signal.get("description_of_the_reports_problem", [])
        prev_descriptions = [raw] if isinstance(raw, str) and raw else (raw or [])

        # News agent is formula-based (ratio/counts) and does not need LLM validation.
        # We only run a minimal deterministic sanity check.
        if agent_name in ("agent_for_news_analysis", "agent_for_twitter_analysis"):
            print(f"{TAG}   {agent_name}: skipping LLM validation (deterministic check only)")
            deterministic_problem = ""
            if not summary:
                deterministic_problem = f"Empty summary in {agent_name} report."
            elif prediction not in (True, False):
                deterministic_problem = f"Prediction must be boolean for {agent_name} report."

            if deterministic_problem:
                print(f"{TAG}   RESULT: PROBLEM FOUND")
                print(f"{TAG}   Problem description: {deterministic_problem}")
                retry_agents.append(agent_name)
                updated_signals[agent_name] = {
                    **signal,
                    "description_of_the_reports_problem": prev_descriptions + [deterministic_problem],
                }
            else:
                print(f"{TAG}   RESULT: OK — deterministic checks passed")
                updated_signals[agent_name] = {
                    **signal,
                    "description_of_the_reports_problem": prev_descriptions,
                }
            continue

        if not reasoning and not summary:
            print(f"{TAG}   {agent_name}: No reasoning/summary found — stub agent, skipping")
            updated_signals[agent_name] = signal
            continue

        pred_label = "HIGHER (True)" if prediction is True else "LOWER (False)"
        confidence = signal.get("confidence", "unknown")
        print(f"{TAG}   Prediction: {pred_label} | Confidence: {confidence}")
        print(f"{TAG}   Reasoning length: {len(reasoning)} chars | Summary length: {len(summary)} chars")
        print(f"{TAG}   Previous feedback iterations: {len(prev_descriptions)}")

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Report from agent '{agent_name}':\n\n"
                f"**reasoning:**\n{reasoning}\n\n"
                f"**summary:**\n{summary}\n\n"
                f"**risks:**\n{risks or '(not specified)'}\n\n"
                f"**prediction:** {pred_label}\n"
                f"**confidence:** {confidence}"
            )),
        ]

        if prev_descriptions:
            history = "\n".join(f"Iteration {i+1}: {d}" for i, d in enumerate(prev_descriptions))
            messages.append(HumanMessage(content=f"History of previous validator feedback:\n{history}"))

        print(f"{TAG}   Calling validator LLM with {len(messages)} messages...")
        result = cast(AgentValidationResult, llm.with_structured_output(AgentValidationResult).invoke(messages))

        if result.has_problem:
            print(f"{TAG}   RESULT: PROBLEM FOUND")
            print(f"{TAG}   Problem description: {result.description[:500]}")
            retry_agents.append(agent_name)
        else:
            print(f"{TAG}   RESULT: OK — report passed validation")

        updated_signals[agent_name] = {
            **signal,
            "prediction": prediction,
            "description_of_the_reports_problem": prev_descriptions + ([result.description] if result.has_problem else []),
        }

    print(f"\n{TAG} Validation complete: {len(retry_agents)} agent(s) need retry: {retry_agents if retry_agents else 'none'}")
    return {"agent_signals": updated_signals, "retry_agents": retry_agents}
