"""Pydantic models for API request/response validation."""

import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    dates: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of dates in YYYY-MM-DD format",
        examples=[["2025-01-20", "2025-01-21"]]
    )
    config_name: Literal["baseline_1d", "baseline_3d", "baseline_5d", "baseline_7d"] = Field(
        ...,
        description="Configuration name for prediction horizon"
    )

    @field_validator("dates")
    @classmethod
    def validate_date_format(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in v:
            if not pattern.match(d):
                raise ValueError(f"Invalid date format: {d}. Expected YYYY-MM-DD")
        return v


class ModelPrediction(BaseModel):
    """Single model's prediction output."""

    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of price increase")


class SinglePrediction(BaseModel):
    """Prediction result for a single date."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    base_model: ModelPrediction = Field(..., description="Base model prediction")
    range_model: ModelPrediction = Field(..., description="Range model prediction")


class PredictionResponse(BaseModel):
    """Response schema for predictions endpoint."""

    config_name: str = Field(..., description="Configuration used")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    requested_dates: list[str] = Field(..., description="Dates requested")
    found_dates: list[str] = Field(..., description="Dates found in data")
    missing_dates: list[str] = Field(..., description="Dates not found in data")
    predictions: list[SinglePrediction] = Field(..., description="List of predictions")


class ConfigInfo(BaseModel):
    """Information about a single configuration."""

    name: str
    horizon_days: int
    ma_window: int
    feature_count: int


class ConfigsResponse(BaseModel):
    """Response schema for configs endpoint."""

    available_configs: list[ConfigInfo]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = "healthy"
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded models per config"
    )
