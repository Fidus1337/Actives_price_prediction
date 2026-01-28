"""FastAPI application entry point."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Set working directory for config/model paths

from api.routers import predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Starting BTC Price Prediction API...")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="BTC Price Direction Prediction API",
    description="""
API for predicting Bitcoin price direction using machine learning models.

## Features
- Predict if BTC price will be higher after N days (1, 3, 5, or 7 days)
- Multiple model types: base model and range model
- Historical date predictions based on CoinGlass data

## Models
- **base_model**: Uses standard market features
- **range_model**: Includes volatility/range features for enhanced prediction

## Configurations
- `baseline_1d`: 1-day prediction horizon
- `baseline_3d`: 3-day prediction horizon
- `baseline_5d`: 5-day prediction horizon
- `baseline_7d`: 7-day prediction horizon
""",
    version="1.0.0",
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
    return {"message": "BTC Price Prediction API", "docs": "/docs"}
