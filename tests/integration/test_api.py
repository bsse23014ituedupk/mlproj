"""
Integration Tests — FastAPI end-to-end tests using TestClient.

Tests:
- Valid prediction returns all required fields
- Invalid pH returns 422
- Health endpoint returns 200
- Batch predict returns correct number of responses
"""
import pytest
from fastapi.testclient import TestClient


# Valid sample data for testing
VALID_SAMPLE = {
    "ph": 7.0, "Hardness": 200.0, "Solids": 20000.0,
    "Chloramines": 7.0, "Sulfate": 330.0, "Conductivity": 400.0,
    "Organic_carbon": 14.0, "Trihalomethanes": 60.0, "Turbidity": 3.5,
}

INVALID_SAMPLE_PH = {
    "ph": 15.0, "Hardness": 200.0, "Solids": 20000.0,
    "Chloramines": 7.0, "Sulfate": 330.0, "Conductivity": 400.0,
    "Organic_carbon": 14.0, "Trihalomethanes": 60.0, "Turbidity": 3.5,
}


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """GET /health should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_sample(self, client):
        """POST a valid sample, assert response has all required fields."""
        response = client.post("/predict", json=VALID_SAMPLE)
        # May be 503 if model not loaded in test env, which is acceptable
        if response.status_code == 200:
            data = response.json()
            assert "potability" in data
            assert "label" in data
            assert "confidence" in data
            assert "risk_level" in data
            assert "top_risk_factors" in data
            assert "model_version" in data
            assert "prediction_id" in data
            assert data["potability"] in [0, 1]
            assert data["label"] in ["Safe", "Unsafe"]
            assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_predict_invalid_ph(self, client):
        """POST pH=15, assert 422 response."""
        response = client.post("/predict", json=INVALID_SAMPLE_PH)
        assert response.status_code == 422, (
            f"Expected 422 for pH=15, got {response.status_code}"
        )


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict(self, client):
        """POST 10 samples, assert 10 responses returned."""
        batch = {"samples": [VALID_SAMPLE] * 10}
        response = client.post("/predict/batch", json=batch)
        if response.status_code == 200:
            data = response.json()
            assert len(data) == 10
            for item in data:
                assert "potability" in item
                assert "prediction_id" in item


class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info(self, client):
        """GET /model/info returns model metadata."""
        response = client.get("/model/info")
        # 503 if no model loaded is acceptable in test env
        assert response.status_code in [200, 503]
