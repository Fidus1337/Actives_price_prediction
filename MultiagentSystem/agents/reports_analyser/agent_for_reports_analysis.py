from multiagent_types import AgentState


CONFIDENCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def compute_confidence_score(
    agent_signals: dict,
    neutral_threshold: int = 1,
) -> tuple[int, str | None, str]:
    """
    Mathematical aggregation of agent predictions.

    Returns (score, direction, breakdown_text).
    - score: sum of weighted predictions (+high=3, +medium=2, +low=1 for HIGHER; negative for LOWER)
    - direction: "LONG" | "SHORT" | None
    - breakdown_text: text breakdown of the calculation
    """
    score = 0
    parts = []

    for name, signal in agent_signals.items():
        # Skip stub agents (agent_c, agent_d)
        if signal.get("summary") is None or not signal.get("reasoning"):
            continue

        confidence = signal.get("confidence", "low")
        weight = CONFIDENCE_WEIGHTS.get(confidence, 1)
        prediction = signal.get("prediction")

        if prediction is True:
            score += weight
            parts.append(f"{name}: HIGHER ({confidence}) -> +{weight}")
        else:
            score -= weight
            parts.append(f"{name}: LOWER ({confidence}) -> -{weight}")

    breakdown = " | ".join(parts) if parts else "No real reports"

    # Determine direction with neutral zone
    if score > neutral_threshold:
        direction = "LONG"
    elif score < -neutral_threshold:
        direction = "SHORT"
    else:
        direction = None

    return score, direction, breakdown


def agent_reports_analyser(state: AgentState):
    TAG = "[reports_analyser]"
    signals = state.get("agent_signals", {})
    threshold = state.get("config", {}).get("neutral_threshold", 1)

    print(f"\n{'='*60}")
    print(f"{TAG} === AGGREGATING FINAL VERDICT ===")
    print(f"{'='*60}")
    print(f"{TAG} Received {len(signals)} agent signals | neutral_threshold={threshold}")

    for name, signal in signals.items():
        pred = signal.get("prediction")
        conf = signal.get("confidence", "?")
        has_summary = bool(signal.get("summary"))
        has_reasoning = bool(signal.get("reasoning"))
        problems = signal.get("description_of_the_reports_problem", [])
        pred_label = "HIGHER" if pred is True else ("LOWER" if pred is False else "N/A")
        print(f"{TAG}   {name}: prediction={pred_label} confidence={conf} has_summary={has_summary} has_reasoning={has_reasoning} validation_issues={len(problems)}")

    score, direction, breakdown = compute_confidence_score(signals, threshold)

    direction_label = direction or "NEUTRAL"
    print(f"{TAG} Score calculation: {breakdown}")
    print(f"{TAG} Final score: {score} (neutral zone: +/-{threshold}) -> {direction_label}")

    reasoning = f"Confidence score: {score} (neutral zone: +/-{threshold}). {breakdown}"
    summary = f"{direction_label} (score={score})"

    print(f"{TAG} Done. Final verdict: {direction_label}")

    return {
        "general_prediction_by_all_reports": direction,
        "general_reports_summary": summary,
        "general_reports_reasoning": reasoning,
        "general_reports_risks": "",
        "confidence_score": score,
    }
