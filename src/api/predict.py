"""
Prediction Logic — converts a WaterSample into a PredictionResponse.

Handles feature engineering, preprocessing, prediction, and SHAP
explanation for each individual sample.
"""
import uuid
import numpy as np

from src.config import FEATURE_COLUMNS
from src.features.feature_engineering import (
    create_engineered_features_array, get_all_feature_names,
)
from src.models.evaluate import compute_single_shap
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_sample(
    sample_dict: dict,
    model,
    pipeline,
    model_version: str = "local",
) -> dict:
    """Generate a prediction for a single water sample.

    Args:
        sample_dict: Dict with the 9 water quality features.
        model: Trained sklearn model.
        pipeline: Fitted preprocessing pipeline.
        model_version: Model version string for traceability.

    Returns:
        Dict matching the PredictionResponse schema.
    """
    # Build feature array in correct column order (9 original features)
    raw_features = np.array([[sample_dict[col] for col in FEATURE_COLUMNS]])

    # Apply feature engineering (adds 3 interaction features = 12 total)
    raw_with_fe = create_engineered_features_array(raw_features)

    # Apply preprocessing pipeline (imputation + clipping + scaling)
    # The pipeline was fitted on 12 features during training.
    processed_with_fe = pipeline.transform(raw_with_fe)

    # Predict
    prediction = int(model.predict(processed_with_fe)[0])
    probabilities = model.predict_proba(processed_with_fe)[0]
    confidence = float(probabilities[prediction])

    # Determine risk level based on probability of being unsafe (class 0)
    unsafe_prob = float(probabilities[0])
    if unsafe_prob >= 0.7:
        risk_level = "High"
    elif unsafe_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Compute per-prediction SHAP explanations
    feature_names = get_all_feature_names()
    top_factors = compute_single_shap(model, processed_with_fe, feature_names)

    return {
        "potability": prediction,
        "label": "Safe" if prediction == 1 else "Unsafe",
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_factors,
        "model_version": model_version,
        "prediction_id": str(uuid.uuid4()),
    }


def predict_batch(
    samples: list[dict],
    model,
    pipeline,
    model_version: str = "local",
) -> list[dict]:
    """Generate predictions for a batch of water samples.

    Args:
        samples: List of dicts with water quality features.
        model: Trained sklearn model.
        pipeline: Fitted preprocessing pipeline.
        model_version: Model version string.

    Returns:
        List of prediction response dicts.
    """
    return [
        predict_sample(s, model, pipeline, model_version)
        for s in samples
    ]
