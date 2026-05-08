"""
Model Registry Module — MLflow Model Registry push/load utilities.

Handles registering the best model to MLflow's Model Registry and
loading it for production inference.
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn

from src.config import MLFLOW_MODEL_NAME, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def register_best_model(run_id: str, model_name: str = MLFLOW_MODEL_NAME) -> str:
    """Register a model from an MLflow run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the model artifact.
        model_name: Registry name for the model.

    Returns:
        Model version string.
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    logger.info(f"Registered model '{model_name}' version {result.version}")
    return result.version


def load_production_model(model_name: str = MLFLOW_MODEL_NAME):
    """Load the production model from MLflow Model Registry.

    Falls back to loading from local joblib if MLflow registry is
    not available (e.g., in Docker without MLflow server).

    Args:
        model_name: Registry name of the model.

    Returns:
        Loaded sklearn model.
    """
    try:
        model = mlflow.sklearn.load_model(
            f"models:/{model_name}/Production"
        )
        logger.info(f"Loaded model from MLflow registry: {model_name}/Production")
        return model
    except Exception as e:
        logger.warning(f"MLflow registry load failed: {e}. Falling back to local.")
        return load_local_model()


def load_local_model():
    """Load the best model from local joblib file.

    Returns:
        Loaded sklearn model.

    Raises:
        FileNotFoundError: If no model file exists.
    """
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def load_preprocessing_pipeline():
    """Load the fitted preprocessing pipeline from disk.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If no pipeline file exists.
    """
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"No pipeline found at {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    logger.info(f"Loaded preprocessing pipeline from {pipeline_path}")
    return pipeline


def load_model_metadata() -> dict:
    """Load model metadata (best params, metrics, rationale).

    Returns:
        Dict with model metadata.
    """
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return {"error": "No metadata file found"}
    with open(meta_path, "r") as f:
        return json.load(f)
