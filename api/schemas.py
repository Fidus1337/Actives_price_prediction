"""Pydantic models for API request/response validation."""

import re
from pydantic import BaseModel, Field, field_validator, model_serializer
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    models: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of model names from Models folder",
        examples=[["base_model_1d", "range_model_3d"]]
    )
    dates: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of dates in YYYY-MM-DD format",
        examples=[["2025-01-20", "2025-01-21"]]
    )
    refresh_dataset: bool = Field(
        default=False,
        description="Force refresh dataset from API before predicting. "
                    "If False, uses cached data (auto-loads on first request)."
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
    spot_price_close: float | None = Field(None, description="BTC spot close price on prediction date")
    range_abs_sma: float | None = Field(None, description="SMA of daily price range in absolute USD terms, window size from model config. Populated for range models only.")
    sma_window: int | None = Field(None, exclude=True)

    @model_serializer(mode="wrap")
    def _serialize(self, handler) -> dict:
        data = handler(self)
        if "range_abs_sma" in data:
            # Hide nullable field entirely for base models / missing values
            if data["range_abs_sma"] is None:
                data.pop("range_abs_sma", None)
            elif self.sma_window is not None:
                data[f"range_abs_sma_{self.sma_window}"] = data.pop("range_abs_sma")
        return data


class ModelPredictionResult(BaseModel):
    """Prediction results for a single model."""

    model_name: str = Field(..., description="Model used for predictions")
    model_type: str = Field(..., description="Model type (base or range)")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    found_dates: list[str] = Field(..., description="Dates found in data")
    missing_dates: list[str] = Field(..., description="Dates not found in data")
    predictions: list[SinglePrediction] = Field(..., description="List of predictions")
    error: str | None = Field(None, description="Error message if model failed")


class PredictionResponse(BaseModel):
    """Response schema for batch predictions endpoint."""

    requested_models: list[str] = Field(..., description="Models requested")
    requested_dates: list[str] = Field(..., description="Dates requested")
    results: list[ModelPredictionResult] = Field(..., description="Predictions per model")


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


class DatasetStatusResponse(BaseModel):
    """Response schema for dataset status endpoint."""

    is_loaded: bool = Field(..., description="Whether dataset is currently loaded in memory")
    last_refreshed_at: str | None = Field(None, description="ISO datetime of last refresh, null if never loaded")
    shape: list[int] | None = Field(None, description="[rows, columns], null if not loaded")

# Схема ожидаемого JSON. 
# Мы повторяем структуру config.json, где есть ключ "runs", содержащий список конфигов.
class TrainConfigRequest(BaseModel):
    runs: List[Dict[str, Any]]

    # Добавляем конфигурацию с примером
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "runs": [
                        {
                            "name": "base_model_1d",
                            "N_DAYS": 1,
                            "threshold": 0.5,
                            "ma_window": 14,
                            "range_feats": [
                                "range_pct",
                                "range_pct_ma14"
                            ],
                            "base_feats": [
                                "sp500__open__diff1__lag15",
                                "futures_open_interest_aggregated_history__close__pct1",
                                "gold__high__diff1",
                                "futures_open_interest_aggregated_history__close__diff1",
                                "futures_funding_rate_history__open__pct1",
                                "futures_top_long_short_position_ratio_history__top_position_long_percent",
                                "gold__volume__diff1__lag1",
                                "gold__low__diff1__lag1",
                                "sp500__open__diff1__lag5",
                                "futures_open_interest_aggregated_stablecoin_history__high__diff1",
                                "gold__open__diff1__lag7",
                                "index_btc_active_addresses__active_address_count",
                                "sp500__close__diff1__lag3",
                                "gold__open__diff1__lag15"
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
class TrainConfigResponse(BaseModel):
    status: str
    message: str
    source: str

    # Опционально: пример, который будет показан в документации
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Training executed successfully.",
                    "source": "custom_json"
                }
            ]
        }
    }