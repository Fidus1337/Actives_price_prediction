"""Multiagent prediction endpoints router."""

import asyncio
import os
from fastapi import APIRouter, Body, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from MultiagentSystem.multiagent_predictions_module import make_prediction_for_last_N_days, add_y_true
from MultiagentSystem.multiagent_system_main import app as multiagent_app
from MultiagentSystem.agents.news_analyser.news_collector import collect_news
from MultiagentSystem.agents.news_analyser.news_archive_database_manipulator import get_latest_date as news_get_latest_date
from MultiagentSystem.agents.economic_calendar_analyser.calendar_collector import collect_calendar_events
from MultiagentSystem.agents.economic_calendar_analyser.economic_calendar_database_manipulator import get_latest_date as calendar_get_latest_date
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_db import get_latest_date as twitter_get_latest_date
from MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline import run_fetch_only as twitter_fetch
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping import (
    check_twitter_auth,
    save_cookies_from_upload,
)
from api.schemas import (
    MultiagentPredictionsRequest,
    MultiagentPredictionsResponse,
    MultiagentSinglePrediction,
    CollectAgentDataRequest,
    CollectAgentDataResponse,
    CollectAgentDataResult,
    AgentsDataStatusResponse,
    TwitterAuthStatusResponse,
    TwitterCookiesUploadRequest,
    TwitterCookiesUploadResponse,
)


router = APIRouter(prefix="/api", tags=["multiagent_predictions"])

_prediction_lock = asyncio.Lock()

_collection_locks: dict[str, asyncio.Lock] = {
    "news_analyser": asyncio.Lock(),
    "economic_calendar_analyser": asyncio.Lock(),
    "twitter_analyser": asyncio.Lock(),
}


@router.post(
    "/multiagent_predictions",
    response_model=MultiagentPredictionsResponse,
    summary="Run multiagent predictions",
    description="Runs the multiagent system for last N eligible dates using request body shaped like multiagent_config.json",
)
async def multiagent_predictions(request: MultiagentPredictionsRequest) -> MultiagentPredictionsResponse:
    """Run multiagent predictions using request config."""
    if _prediction_lock.locked():
        raise HTTPException(status_code=409, detail="Multiagent prediction is already running")

    async with _prediction_lock:
        try:
            config = request.model_dump(exclude={"n_last_dates"})
            results_df = await run_in_threadpool(
                make_prediction_for_last_N_days,
                multiagent_app,
                config,
                request.n_last_dates,
            )
            results_df = add_y_true(results_df, request.horizon)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to run multiagent predictions: {exc}") from exc

    _DIRECTION_MAP = {"LONG": 1, "SHORT": 0}
    predictions: list[MultiagentSinglePrediction] = []

    for _, row in results_df.iterrows():
        raw_date = row["forecast_start_date"]
        if hasattr(raw_date, "strftime"):
            date_value = raw_date.strftime("%Y-%m-%d")
        else:
            date_value = str(raw_date)

        predictions.append(
            MultiagentSinglePrediction(
                date=date_value,
                y_true=_DIRECTION_MAP.get(row.get("y_true")),
                y_prediction=_DIRECTION_MAP.get(row.get("y_predict")),
                confidence_score=row.get("y_predict_confidence"),
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
    "twitter_analyser": twitter_fetch,
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
        lock = _collection_locks[agent_name]
        if lock.locked():
            raise HTTPException(status_code=409, detail=f"Collection for '{agent_name}' is already running")
        async with lock:
            try:
                if agent_name == "twitter_analyser":
                    raw = await run_in_threadpool(
                        twitter_fetch,
                        request.twitter_since_date,
                        request.twitter_until_date,
                        request.twitter_authors,
                    )
                else:
                    raw = await run_in_threadpool(_AGENT_COLLECTORS[agent_name])
                if agent_name == "twitter_analyser":
                    stats = {
                        "before": raw.get("db_total", 0) - raw.get("tweets_inserted", 0),
                        "fetched": raw.get("tweets_fetched", 0),
                        "new": raw.get("tweets_inserted", 0),
                        "after": raw.get("db_total", 0),
                        "date_range": raw.get("db_range"),
                    }
                else:
                    stats = raw
                results.append(CollectAgentDataResult(agent=agent_name, **stats))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Collection failed for '{agent_name}': {exc}") from exc

    return CollectAgentDataResponse(results=results)


@router.get(
    "/agents/data-status",
    response_model=AgentsDataStatusResponse,
    summary="Last fetched date per agent archive",
    description="Returns MAX(date) from each agent's SQLite archive. Useful to check how fresh the stored data is before triggering collection.",
)
async def agents_data_status() -> AgentsDataStatusResponse:
    try:
        news_date, calendar_date, twitter_date = await run_in_threadpool(
            lambda: (
                news_get_latest_date(),
                calendar_get_latest_date(),
                twitter_get_latest_date(),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query agent archives: {exc}") from exc

    return AgentsDataStatusResponse(
        news_analyser=news_date,
        economic_calendar_analyser=calendar_date,
        twitter_analyser=twitter_date,
    )


@router.get(
    "/agents/twitter-auth-status",
    response_model=TwitterAuthStatusResponse,
    summary="Twitter session / cookie health check",
    description=(
        "Lightweight check (no Chrome launch). Inspects twitter_cookies.json for "
        "session cookies (auth_token, ct0) and verifies credentials are configured. "
        "If relogin_required=true, run: "
        "python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login"
    ),
)
async def twitter_auth_status() -> TwitterAuthStatusResponse:
    try:
        info = await run_in_threadpool(check_twitter_auth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check Twitter auth: {exc}") from exc

    relogin_required = not info["session_cookies_ok"] or not info["credentials_configured"]

    return TwitterAuthStatusResponse(
        **info,
        relogin_required=relogin_required,
    )


@router.post(
    "/agents/twitter-upload-cookies",
    response_model=TwitterCookiesUploadResponse,
    summary="Upload Twitter cookies for re-login without stopping the API",
    description=(
        "Replaces twitter_cookies.json with the uploaded cookies. "
        "Use when session expires and manual re-login via GUI is unavailable. "
        "How to get cookies: open x.com in your browser → DevTools (F12) → "
        "Application → Cookies → x.com, then export with EditThisCookie extension "
        "or copy manually. Must include auth_token and ct0."
    ),
)
async def twitter_upload_cookies(
    request: TwitterCookiesUploadRequest = Body(...),
) -> TwitterCookiesUploadResponse:
    expected_key = os.getenv("TWITTER_UPLOAD_KEY", "")
    if not expected_key or request.upload_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid upload key")
    cookies = request.cookies
    try:
        info = await run_in_threadpool(save_cookies_from_upload, cookies)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save cookies: {exc}") from exc

    return TwitterCookiesUploadResponse(
        saved=info["cookies_count"],
        session_cookies_ok=info["session_cookies_ok"],
        cookies_path=info["cookies_path"],
    )
