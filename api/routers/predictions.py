"""Prediction endpoints router."""

import json
import os
import sys
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set working directory to project root (Predictor uses relative paths for models)
os.chdir(PROJECT_ROOT)

from Predictor import Predictor
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    SinglePrediction,
    ModelsResponse,
    ModelInfo,
    ModelMetrics,
    HealthResponse,
)

router = APIRouter(prefix="/api/v1", tags=["predictions"])

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
    description="Returns predictions for specified dates using the selected model",
)
async def get_predictions(request: PredictionRequest) -> PredictionResponse:
    """
    Get predictions for specified dates using the selected model.

    The prediction indicates whether BTC price will be higher after N days.
    """
    # Validate model_name
    if not validate_model_name(request.model_name):
        available = get_available_models()
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found. "
            f"Available models: {available}",
        )

    try:
        # Get or create Predictor
        predictor = get_predictor(request.model_name)

        # Get predictions
        results = predictor.predict_by_dates(request.dates)

        # Build response
        found_dates = [r.date for r in results]
        missing_dates = list(set(request.dates) - set(found_dates))

        predictions = [
            SinglePrediction(
                date=r.date,
                prediction=r.prediction,
                probability=round(r.probability, 6),
            )
            for r in results
        ]

        return PredictionResponse(
            model_name=request.model_name,
            model_type=predictor.model_type,
            horizon_days=predictor.n_days,
            requested_dates=request.dates,
            found_dates=found_dates,
            missing_dates=missing_dates,
            predictions=predictions,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model files not found for {request.model_name}: {e}",
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Prediction error: {tb}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
                auc=metrics_data.get("auc", 0.0),
                accuracy=metrics_data.get("acc", 0.0),
                precision=metrics_data.get("precision", 0.0),
                recall=metrics_data.get("recall", 0.0),
                f1=metrics_data.get("f1", 0.0),
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
