"""Multiagent prediction endpoints router."""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from MultiagentSystem.multiagent_predictions_module import make_prediction_for_last_N_days
from MultiagentSystem.agents.news_analyser.news_collector import collect_news
from MultiagentSystem.agents.economic_calendar_analyser.calendar_collector import collect_calendar_events
from api.schemas import (
    MultiagentPredictionsRequest,
    MultiagentPredictionsResponse,
    MultiagentSinglePrediction,
    CollectAgentDataRequest,
    CollectAgentDataResponse,
    CollectAgentDataResult,
)


router = APIRouter(prefix="/api", tags=["multiagent_predictions"])


@router.post(
    "/multiagent_predictions",
    response_model=MultiagentPredictionsResponse,
    summary="Run multiagent predictions",
    description="Runs the multiagent system for last N eligible dates using request body shaped like multiagent_config.json",
)
async def multiagent_predictions(request: MultiagentPredictionsRequest) -> MultiagentPredictionsResponse:
    """Run multiagent predictions using request config."""
    try:
        config = request.model_dump(exclude={"n_last_dates"})
        results_df = await run_in_threadpool(
            make_prediction_for_last_N_days,
            config,
            request.n_last_dates,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run multiagent predictions: {exc}") from exc

    predictions: list[MultiagentSinglePrediction] = []
    horizon_col = f"y_up_{int(request.horizon)}d"

    for _, row in results_df.iterrows():
        raw_date = row["date"]
        if hasattr(raw_date, "strftime"):
            date_value = raw_date.strftime("%Y-%m-%d")
        else:
            date_value = str(raw_date)

        raw_pred = row["y_predictions"]
        prediction_value = None if pd.isna(raw_pred) else int(raw_pred)

        predictions.append(
            MultiagentSinglePrediction(
                date=date_value,
                y_true=int(row[horizon_col]),
                y_prediction=prediction_value,
                confidence_score=row["confidence_score"],
            )
        )

    return MultiagentPredictionsResponse(
        requested_forecast_start_date=request.forecast_start_date,
        requested_horizon=request.horizon,
        requested_n_last_dates=request.n_last_dates,
        rows_returned=len(predictions),
        predictions=predictions,
    )


_AGENT_COLLECTORS = {
    "news_analyser": collect_news,
    "economic_calendar_analyser": collect_calendar_events,
}


@router.post(
    "/system/collect_agent_data",
    response_model=CollectAgentDataResponse,
    summary="Collect latest data for news and calendar agents",
    description="Triggers incremental data collection for the specified agents, appending new records to their SQLite archives.",
)
async def collect_agent_data(request: CollectAgentDataRequest) -> CollectAgentDataResponse:
    unknown = set(request.agents) - _AGENT_COLLECTORS.keys()
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown agents: {sorted(unknown)}")

    results = []
    for agent_name in request.agents:
        try:
            stats = await run_in_threadpool(_AGENT_COLLECTORS[agent_name])
            results.append(CollectAgentDataResult(agent=agent_name, **stats))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Collection failed for '{agent_name}': {exc}") from exc

    return CollectAgentDataResponse(results=results)
