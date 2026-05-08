"""
Health and Readiness Endpoints — for container orchestration and monitoring.

- /health: Simple liveness check (is the process running?)
- /ready: Readiness check (is the model loaded and ready to serve?)
"""
from src.api.schemas import HealthResponse, ReadinessResponse


def get_health_status(model_loaded: bool) -> HealthResponse:
    """Return health status.

    Args:
        model_loaded: Whether the model is currently loaded.

    Returns:
        HealthResponse with status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
    )


def get_readiness_status(
    model_loaded: bool,
    metadata: dict,
) -> ReadinessResponse:
    """Return readiness status with model metadata.

    Args:
        model_loaded: Whether the model is loaded.
        metadata: Model metadata dict.

    Returns:
        ReadinessResponse with model details.
    """
    return ReadinessResponse(
        status="ready" if model_loaded else "not_ready",
        model_loaded=model_loaded,
        model_name=metadata.get("best_model", "unknown"),
        test_auc=metadata.get("test_roc_auc"),
        feature_count=len(metadata.get("feature_names", [])),
    )
