"""Prediction endpoints router."""

import json
import os
import sys
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

from typing import Any, Dict, List, Optional  # <--- Нужно для аннотации типов
from pydantic import BaseModel                # <--- Нужно для Pydantic моделей

# Добавляем Body в этот список:
from fastapi import APIRouter, HTTPException, Body 
from fastapi.concurrency import run_in_threadpool

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.concurrency import run_in_threadpool
from Models_builder_pipeline import run_all_configs

# Set working directory to project root (Predictor uses relative paths for models)
os.chdir(PROJECT_ROOT)

from Predictor import Predictor
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelPredictionResult,
    SinglePrediction,
    ModelsResponse,
    ModelInfo,
    ModelMetrics,
    HealthResponse,
    TrainConfigRequest,
    TrainConfigResponse
)

router = APIRouter(prefix="/api", tags=["predictions"])

# Cache for Predictor instances (one per model)
_predictor_cache: dict[str, Predictor] = {}

MODELS_DIR = PROJECT_ROOT / "Models"


def get_predictor(model_name: str) -> Predictor:
    """Get or create a Predictor instance for the given model."""
    
    if model_name not in _predictor_cache:
        _predictor_cache[model_name] = Predictor(
            config_name=model_name,
            config_path=str(PROJECT_ROOT / "config.json"),
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
    config_path = PROJECT_ROOT / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
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
    response_model=PredictionResponse,
    summary="Get BTC price direction predictions",
    description="Returns predictions for specified dates using the selected models",
)
async def get_predictions(request: PredictionRequest) -> PredictionResponse:
    """
    Get predictions for specified dates using the selected models.

    Each model predicts whether BTC price will be higher after N days
    for every requested date.
    """
    available = get_available_models()
    results: list[ModelPredictionResult] = []

    # Refresh shared data: only when explicitly requested or on first load
    from api.main import shared_data_cache
    data_refreshed = False
    if shared_data_cache is not None:
        if request.refresh_dataset:
            shared_data_cache.refresh()
            data_refreshed = True
        elif shared_data_cache._base_df is None:
            # No data yet — force initial load
            shared_data_cache.refresh()
            data_refreshed = True

    for model_name in request.models:
        if not validate_model_name(model_name):
            results.append(ModelPredictionResult(
                model_name=model_name,
                model_type="unknown",
                horizon_days=0,
                found_dates=[],
                missing_dates=request.dates,
                predictions=[],
                error=f"Model '{model_name}' not found. Available: {available}",
            ))
            continue

        try:
            predictor = get_predictor(model_name)
            if data_refreshed or predictor._prepared_df is None:
                predictor.invalidate_cache()
            preds = predictor.predict_by_dates(request.dates)

            found_dates = [r.date for r in preds]
            missing_dates = list(set(request.dates) - set(found_dates))

            predictions = [
                SinglePrediction(
                    date=r.date,
                    prediction=r.prediction,
                    probability=round(r.probability, 6),
                    spot_price=r.spot_price,
                )
                for r in preds
            ]

            results.append(ModelPredictionResult(
                model_name=model_name,
                model_type=predictor.model_type,
                horizon_days=predictor.n_days,
                found_dates=found_dates,
                missing_dates=missing_dates,
                predictions=predictions,
            ))

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Prediction error for {model_name}: {tb}")
            results.append(ModelPredictionResult(
                model_name=model_name,
                model_type="unknown",
                horizon_days=0,
                found_dates=[],
                missing_dates=request.dates,
                predictions=[],
                error=str(e),
            ))

    return PredictionResponse(
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

        models.append(ModelInfo(
            name=name,
            model_type=model_type,
            horizon_days=horizon_days,
            feature_count=feature_count,
            metrics=metrics,
        ))

    return ModelsResponse(available_models=models)

@router.post(
    "/system/train_models",
    summary="Run configs and reload models",
    description="Clears models cache -> Trains models. If JSON body is provided, uses it. Otherwise, loads from config.json.",
    response_model=TrainConfigResponse  # <--- ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ
)
async def train_models(
    # Body(None) делает тело запроса необязательным.
    # Если JSON пришел, он попадет в переменную config_payload.
    config_payload: Optional[TrainConfigRequest] = Body(None)
):
    """
    1. Очищает кеш загруженных моделей.
    2. Если передан JSON, берет конфиги из него.
    3. Если JSON нет, берет конфиги из файла config.json.
    4. Запускает обучение.
    """
    global _predictor_cache
    
    # Подготовка конфига
    custom_runs = None
    if config_payload:
        custom_runs = config_payload.runs
        print(f"Received custom config with {len(custom_runs)} runs.")
    else:
        print("No custom config provided, using default 'config.json'.")

    # 1. Очищаем кеш моделей и shared data cache
    print("Clearing model cache and shared data cache...")
    _predictor_cache.clear()
    Predictor.clear_shared_cache()

    # 2. Запускаем тяжелую функцию
    try:
        print("Starting train_models...")

        await run_in_threadpool(
            run_all_configs,
            "config.json",
            custom_runs
        )

        # 3. Перезагружаем shared data cache после тренировки
        from api.main import shared_data_cache
        if shared_data_cache is not None:
            print("Refreshing shared data cache after training...")
            await run_in_threadpool(shared_data_cache.preload)

        return {
            "status": "success",
            "message": "Training executed successfully.",
            "source": "custom_json" if custom_runs else "file_config"
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error running configs: {tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run configs: {str(e)}"
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

