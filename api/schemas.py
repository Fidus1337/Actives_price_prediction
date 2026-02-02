"""Pydantic models for API request/response validation."""

import re
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    model_name: str = Field(
        ...,
        description="Model name from Models folder (e.g., 'base_model_1d', 'range_model_3d')",
        examples=["base_model_1d", "range_model_1d"]
    )
    dates: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of dates in YYYY-MM-DD format",
        examples=[["2025-01-20", "2025-01-21"]]
    )

    @field_validator("dates")
    @classmethod
    def validate_date_format(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in v:
            if not pattern.match(d):
                raise ValueError(f"Invalid date format: {d}. Expected YYYY-MM-DD")
        return v


class SinglePrediction(BaseModel):
    """Prediction result for a single date."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=down, 1=up)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of price increase")


class PredictionResponse(BaseModel):
    """Response schema for predictions endpoint."""

    model_name: str = Field(..., description="Model used for predictions")
    model_type: str = Field(..., description="Model type (base or range)")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    requested_dates: list[str] = Field(..., description="Dates requested")
    found_dates: list[str] = Field(..., description="Dates found in data")
    missing_dates: list[str] = Field(..., description="Dates not found in data")
    predictions: list[SinglePrediction] = Field(..., description="List of predictions")


class ModelMetrics(BaseModel):
    """Quality metrics for a model."""

    auc: float = Field(..., description="Area Under ROC Curve")
    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 score")
    threshold: float = Field(default=0.5, description="Classification threshold")


class ModelInfo(BaseModel):
    """Information about a single model."""

    name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (base or range)")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    feature_count: int = Field(..., description="Number of features used")
    metrics: ModelMetrics | None = Field(None, description="Model quality metrics")


class ModelsResponse(BaseModel):
    """Response schema for models endpoint."""

    available_models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = "healthy"
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded models"
    )
