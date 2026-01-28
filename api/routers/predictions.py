"""Prediction endpoints router."""

import json
import os
import sys
import traceback
from functools import lru_cache
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
    ModelPrediction,
    ConfigsResponse,
    ConfigInfo,
    HealthResponse,
)

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Cache for Predictor instances (one per config)
_predictor_cache: dict[str, Predictor] = {}


def get_predictor(config_name: str) -> Predictor:
    """Get or create a Predictor instance for the given config."""
    if config_name not in _predictor_cache:
        _predictor_cache[config_name] = Predictor(
            config_name=config_name,
            config_path=str(PROJECT_ROOT / "config.json"),
            env_path=str(PROJECT_ROOT / "dev.env"),
        )
    return _predictor_cache[config_name]


def get_loaded_configs() -> dict[str, bool]:
    """Get status of loaded Predictors."""
    return {name: True for name in _predictor_cache}


@lru_cache(maxsize=1)
def get_available_configs() -> list[dict]:
    """Load and cache available configurations from config.json."""
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("runs", [])


def validate_config_name(config_name: str) -> bool:
    """Check if config_name exists in available configs."""
    configs = get_available_configs()
    return any(c.get("name") == config_name for c in configs)


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Get BTC price direction predictions",
    description="Returns predictions for whether BTC price will be higher after N days",
)
async def get_predictions(request: PredictionRequest) -> PredictionResponse:
    """
    Get predictions for specified dates using the selected configuration.

    The prediction indicates whether BTC price will be higher after N days
    (where N depends on the config: 1d, 3d, 5d, or 7d).
    """
    # Validate config_name
    if not validate_config_name(request.config_name):
        raise HTTPException(
            status_code=404,
            detail=f"Configuration '{request.config_name}' not found. "
            "Use /api/v1/configs to see available options.",
        )

    try:
        # Get or create Predictor (lazy loading with cache)
        predictor = get_predictor(request.config_name)

        # Get predictions
        results = predictor.predict_by_dates(request.dates)

        # Build response
        found_dates = [r.date for r in results]
        missing_dates = list(set(request.dates) - set(found_dates))

        predictions = [
            SinglePrediction(
                date=r.date,
                base_model=ModelPrediction(
                    prediction=r.base_model_pred,
                    probability=round(r.base_model_proba, 6),
                ),
                range_model=ModelPrediction(
                    prediction=r.range_model_pred,
                    probability=round(r.range_model_proba, 6),
                ),
            )
            for r in results
        ]

        return PredictionResponse(
            config_name=request.config_name,
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
            detail=f"Model files not found for {request.config_name}. Run training first.",
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Prediction error: {tb}")  # Log to console
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get(
    "/configs",
    response_model=ConfigsResponse,
    summary="List available configurations",
    description="Returns all available prediction configurations with their parameters",
)
async def list_configs() -> ConfigsResponse:
    """Get list of available prediction configurations."""
    configs = get_available_configs()

    return ConfigsResponse(
        available_configs=[
            ConfigInfo(
                name=c["name"],
                horizon_days=c["N_DAYS"],
                ma_window=c.get("ma_window", 14),
                feature_count=len(c.get("base_feats", [])),
            )
            for c in configs
        ]
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
        models_loaded=get_loaded_configs(),
    )
