"""
FastAPI Application Entry Point — serves water potability predictions.

Endpoints:
- GET  /health       → liveness check
- GET  /ready        → readiness with model metadata
- POST /predict      → single sample prediction
- POST /predict/batch → batch predictions
- GET  /model/info   → feature importances and selection rationale
- GET  /docs         → auto-generated Swagger UI (built-in)
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    WaterSample, PredictionResponse, BatchPredictionRequest,
    HealthResponse, ReadinessResponse, ModelInfoResponse,
)
from src.api.predict import predict_sample, predict_batch
from src.api.health import get_health_status, get_readiness_status
from src.models.registry import (
    load_local_model, load_preprocessing_pipeline, load_model_metadata,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cached model and pipeline
_model = None
_pipeline = None
_metadata: dict = {}
_model_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler.

    Loads the model and preprocessing pipeline at startup,
    releases resources at shutdown.
    """
    global _model, _pipeline, _metadata, _model_loaded
    try:
        logger.info("Loading model and pipeline at startup...")
        _model = load_local_model()
        _pipeline = load_preprocessing_pipeline()
        _metadata = load_model_metadata()
        _model_loaded = True
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        _model_loaded = False

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down API...")
    _model = None
    _pipeline = None


app = FastAPI(
    title="Water Potability Classifier API",
    description=(
        "Production-grade API for predicting water potability using "
        "ML models with SHAP-based explanations. Trained on 9 water "
        "quality indicators from WHO/EPA guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — is the service running?"""
    return get_health_status(_model_loaded)


@app.get("/ready", response_model=ReadinessResponse)
async def ready():
    """Readiness check — is the model loaded and ready to serve?"""
    return get_readiness_status(_model_loaded, _metadata)


@app.post("/predict", response_model=PredictionResponse)
async def predict(sample: WaterSample):
    """Predict potability for a single water sample.

    Returns prediction with confidence, risk level, and top 3
    SHAP-based risk factors explaining the prediction.
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict_sample(
        sample.model_dump(),
        _model,
        _pipeline,
        model_version=_metadata.get("best_model", "local"),
    )
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """Predict potability for a batch of water samples.

    Returns a list of predictions, one per input sample.
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    samples = [s.model_dump() for s in request.samples]
    results = predict_batch(
        samples, _model, _pipeline,
        model_version=_metadata.get("best_model", "local"),
    )
    return [PredictionResponse(**r) for r in results]


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata including feature importances and rationale."""
    if not _metadata:
        raise HTTPException(status_code=503, detail="No model metadata")
    return ModelInfoResponse(**_metadata)
