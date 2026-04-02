"""Classic ML prediction endpoints router."""

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.concurrency import run_in_threadpool

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from Models_builder_pipeline import train_all_models_from_configs

# Set working directory to project root (Predictor uses relative paths for models)
os.chdir(PROJECT_ROOT)

from PredictWithModel.Predictor import Predictor
from api.schemas import (
    ClassicML_PredictionResponse,
    ClassicML_PredictionRequest,
    ClassicML_ModelPredictionResult,
    ClassicML_SinglePrediction,
    ModelsResponse,
    ModelInfo,
    ModelMetrics,
    HealthResponse,
    DatasetStatusResponse,
    TrainConfigRequest,
    TrainConfigResponse,
)

router = APIRouter(prefix="/api", tags=["classic_ml_predictions"])

_train_lock = asyncio.Lock()
_dataset_refresh_lock = asyncio.Lock()

# Cache for Predictor instances (one per model)
_predictor_cache: dict[str, Predictor] = {}

MODELS_DIR = PROJECT_ROOT / "Models"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"


def get_predictor(model_name: str) -> Predictor:
    """Get or create a Predictor instance for the given model."""

    if model_name not in _predictor_cache:
        _predictor_cache[model_name] = Predictor(
            config_name=model_name,
            config_path=str(CONFIG_PATH),
            env_path=str(PROJECT_ROOT / "dev.env"),
        )
    return _predictor_cache[model_name]


def get_loaded_models() -> dict[str, bool]:
    """Get status of loaded Predictors."""
    return {name: True for name in _predictor_cache}


def get_available_models() -> list[str]:
    """Get list of available model names from Models folder."""
    if not MODELS_DIR.exists():
        return []
    return [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]


def load_model_metrics(model_name: str) -> dict | None:
    """Load metrics JSON for a model."""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        return None

    # Find metrics file (pattern: metrics_{type}_{name}.json)
    for f in model_dir.glob("metrics_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return None


def get_model_config(model_name: str) -> dict | None:
    """Get config for a model from config.json."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        for run in config.get("runs", []):
            if run.get("name") == model_name:
                return run
    except Exception:
        pass
    return None


def validate_model_name(model_name: str) -> bool:
    """Check if model exists in Models folder."""
    model_dir = MODELS_DIR / model_name
    return model_dir.exists() and model_dir.is_dir()


@router.post(
    "/predictions",
    response_model=ClassicML_PredictionResponse,
    summary="Get BTC price direction predictions",
    description="Returns predictions for specified dates using the selected models",
)
async def get_predictions(request: ClassicML_PredictionRequest) -> ClassicML_PredictionResponse:
    """
    Get predictions for specified dates using the selected models.

    Each model predicts whether BTC price will be higher after N days
    for every requested date.
    """
    available = get_available_models()
    results: list[ClassicML_ModelPredictionResult] = []

    # Refresh shared data: only when explicitly requested or on first load
    from api.main import shared_data_cache

    data_refreshed = False
    if shared_data_cache is not None:
        if request.refresh_dataset or shared_data_cache._base_df is None:
            if _dataset_refresh_lock.locked():
                raise HTTPException(status_code=409, detail="Dataset refresh is already running")
            async with _dataset_refresh_lock:
                await run_in_threadpool(shared_data_cache.refresh)
            data_refreshed = True

    for model_name in request.models:
        if not validate_model_name(model_name):
            results.append(
                ClassicML_ModelPredictionResult(
                    model_name=model_name,
                    model_type="unknown",
                    horizon_days=0,
                    found_dates=[],
                    missing_dates=request.dates,
                    predictions=[],
                    error=f"Model '{model_name}' not found. Available: {available}",
                )
            )
            continue

        try:
            predictor = get_predictor(model_name)
            if data_refreshed or predictor._prepared_df is None:
                predictor.invalidate_cache()
            preds = predictor.predict_by_dates(request.dates)

            found_dates = [r.date for r in preds]
            missing_dates = list(set(request.dates) - set(found_dates))

            predictions = [
                ClassicML_SinglePrediction(
                    date=r.date,
                    prediction=r.prediction,
                    probability=round(r.probability, 6),
                    spot_price_close=r.spot_price,
                    range_sma=r.range_sma,
                    sma_window=r.sma_window,
                )
                for r in preds
            ]

            results.append(
                ClassicML_ModelPredictionResult(
                    model_name=model_name,
                    model_type=predictor.model_type,
                    horizon_days=predictor.n_days,
                    found_dates=found_dates,
                    missing_dates=missing_dates,
                    predictions=predictions,
                )
            )

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Prediction error for {model_name}: {tb}")
            results.append(
                ClassicML_ModelPredictionResult(
                    model_name=model_name,
                    model_type="unknown",
                    horizon_days=0,
                    found_dates=[],
                    missing_dates=request.dates,
                    predictions=[],
                    error=str(e),
                )
            )

    return ClassicML_PredictionResponse(
        requested_models=request.models,
        requested_dates=request.dates,
        results=results,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Returns all available models with their parameters and quality metrics",
)
async def list_models() -> ModelsResponse:
    """Get list of available models with metrics."""
    model_names = get_available_models()
    models = []

    for name in sorted(model_names):
        # Determine model type
        model_type = "range" if "range" in name else "base"

        # Get config
        config = get_model_config(name)
        horizon_days = config.get("N_DAYS", 1) if config else 1
        feature_count = len(config.get("base_feats", [])) if config else 0
        if model_type == "range" and config:
            feature_count += len(config.get("range_feats", []))

        # Get metrics
        metrics_data = load_model_metrics(name)
        metrics = None
        if metrics_data:
            metrics = ModelMetrics(
                auc=metrics_data.get("cv_avg_auc", 0.0),
                accuracy=metrics_data.get("cv_avg_acc", 0.0),
                precision=metrics_data.get("cv_avg_precision", 0.0),
                recall=metrics_data.get("cv_avg_recall", 0.0),
                f1=metrics_data.get("cv_avg_f1", 0.0),
                threshold=metrics_data.get("thr", 0.5),
            )

        models.append(
            ModelInfo(
                name=name,
                model_type=model_type,
                horizon_days=horizon_days,
                feature_count=feature_count,
                metrics=metrics,
            )
        )

    return ModelsResponse(available_models=models)


@router.post(
    "/system/train_classic_ml_models",
    summary="Run configs and reload models",
    description="Clears models cache -> Trains models. If JSON body is provided, uses it. Otherwise, loads from config.json.",
    response_model=TrainConfigResponse,
)
async def train_classic_ml_models(
    config_payload: Optional[TrainConfigRequest] = Body(None),
):
    """
    1. Clear loaded model cache.
    2. Use body config when provided.
    3. Otherwise, use file config.json.
    4. Run training.
    """
    global _predictor_cache

    if _train_lock.locked():
        raise HTTPException(status_code=409, detail="Model training is already running")

    async with _train_lock:
        # Prepare config
        custom_runs = None
        if config_payload:
            custom_runs = config_payload.runs
            print(f"Received custom config with {len(custom_runs)} runs.")
        else:
            print("No custom config provided, using default 'config.json'.")

        # 1. Clear model cache and shared data cache
        print("Clearing model cache and shared data cache...")
        _predictor_cache.clear()
        Predictor.clear_shared_cache()

        # 2. Run heavy function
        try:
            print("Starting train_classic_ml_models...")
            await run_in_threadpool(train_all_models_from_configs, str(CONFIG_PATH), custom_runs)

            # 3. Reload shared data cache after training
            from api.main import shared_data_cache

            if shared_data_cache is not None:
                print("Refreshing shared data cache after training...")
                await run_in_threadpool(shared_data_cache.refresh)

            return {
                "status": "success",
                "message": "Training executed successfully.",
                "source": "custom_json" if custom_runs else "file_config",
            }
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error running configs: {tb}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to run configs: {str(e)}",
            ) from e


@router.get(
    "/dataset-status",
    response_model=DatasetStatusResponse,
    summary="Dataset status",
    description="Returns the last refresh time and shape of the cached dataset",
)
async def dataset_status() -> DatasetStatusResponse:
    """Check when the shared dataset was last refreshed."""
    from api.main import shared_data_cache

    if shared_data_cache is None or shared_data_cache._base_df is None:
        return DatasetStatusResponse(is_loaded=False, last_refreshed_at=None, shape=None)

    return DatasetStatusResponse(
        is_loaded=True,
        last_refreshed_at=datetime.fromtimestamp(shared_data_cache._fetched_at).isoformat(),
        shape=list(shared_data_cache._base_df.shape),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and model loading status",
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=get_loaded_models(),
    )
