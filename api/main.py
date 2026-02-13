"""FastAPI application entry point."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Set working directory for config/model paths

from api.routers import predictions
from shared_data_cache import SharedBaseDataCache
from Predictor import Predictor

# Module-level reference for access from routers
shared_data_cache: SharedBaseDataCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global shared_data_cache

    print("Starting BTC Price Prediction API...")

    # Load API key and create shared data cache
    load_dotenv(PROJECT_ROOT / "dev.env")
    api_key = os.getenv("COINGLASS_API_KEY")

    if api_key:
        shared_data_cache = SharedBaseDataCache(api_key=api_key)
        Predictor.set_shared_cache(shared_data_cache)
        print("Shared data cache initialized. Fetching dataset...")
        shared_data_cache.refresh()
    else:
        print("Warning: COINGLASS_API_KEY not found. Predictions will fail.")

    yield

    print("Shutting down API...")
    if shared_data_cache is not None:
        shared_data_cache.clear()


app = FastAPI(
    title="BTC Price Direction Prediction API",
    description="""
API for predicting Bitcoin price direction using machine learning models.

## Endpoints

### POST /api/v1/predictions
Get predictions for specific dates using a selected model.

### GET /api/v1/models
List all available models with their quality metrics (AUC, accuracy, precision, recall, F1).

## Models
Models are stored in the `Models/` folder:
- **base_model_Xd**: Uses standard market features
- **range_model_Xd**: Includes volatility/range features

Where X is the prediction horizon (1, 3, 5, or 7 days).

## Example
```python
import requests

# Get predictions
response = requests.post(
    "http://localhost:8000/api/v1/predictions",
    json={
        "model_name": "base_model_1d",
        "dates": ["2025-01-20", "2025-01-21"]
    }
)
```
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to documentation."""
    return {"message": "BTC Price Prediction API v2.0", "docs": "/docs"}
