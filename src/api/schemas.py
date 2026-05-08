"""
Pydantic Schemas — request/response validation models for the FastAPI app.

All fields have strict validation constraints based on physical water
quality bounds to catch impossible inputs before prediction.
"""
import uuid
from pydantic import BaseModel, Field
from typing import Optional


class WaterSample(BaseModel):
    """Input schema for a single water quality sample.

    Each field corresponds to a measured water quality parameter with
    validation constraints based on WHO/EPA guidelines.
    """
    ph: float = Field(..., ge=0.0, le=14.0, description="pH level (WHO: 6.5-8.5)")
    Hardness: float = Field(..., gt=0, description="Calcium carbonate mg/L")
    Solids: float = Field(..., gt=0, description="Total dissolved solids ppm")
    Chloramines: float = Field(..., ge=0, description="Chloramines ppm")
    Sulfate: float = Field(..., ge=0, description="Sulfate mg/L")
    Conductivity: float = Field(..., gt=0, description="Electrical conductivity μS/cm")
    Organic_carbon: float = Field(..., ge=0, description="Organic carbon ppm")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes μg/L")
    Turbidity: float = Field(..., gt=0, description="Turbidity NTU")

    model_config = {"json_schema_extra": {
        "examples": [{
            "ph": 7.0, "Hardness": 200.0, "Solids": 20000.0,
            "Chloramines": 7.0, "Sulfate": 330.0, "Conductivity": 400.0,
            "Organic_carbon": 14.0, "Trihalomethanes": 60.0, "Turbidity": 3.5,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema for a single prediction.

    Includes the prediction, confidence, risk assessment, and
    SHAP-based explanations for interpretability.
    """
    potability: int                          # 0 or 1
    label: str                               # "Safe" or "Unsafe"
    confidence: float                        # Probability of predicted class
    risk_level: str                          # "Low" / "Medium" / "High"
    top_risk_factors: list[str]              # Top 3 SHAP contributors
    model_version: str                       # Model identifier
    prediction_id: str                       # UUID for tracing


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    samples: list[WaterSample]


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool


class ReadinessResponse(BaseModel):
    """Response schema for readiness endpoint."""
    status: str
    model_loaded: bool
    model_name: str
    test_auc: Optional[float] = None
    feature_count: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""
    best_model: str
    test_roc_auc: Optional[float] = None
    test_f1_macro: Optional[float] = None
    best_params: Optional[dict] = None
    feature_names: Optional[list[str]] = None
    selection_rationale: Optional[str] = None
