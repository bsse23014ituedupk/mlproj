"""
Unit Tests — Pydantic schemas.

Tests:
- Invalid pH (>14) raises 422 Unprocessable Entity
- Valid samples pass validation
- Missing required fields raise validation errors
"""
import pytest
from pydantic import ValidationError
from src.api.schemas import WaterSample, PredictionResponse


class TestWaterSampleSchema:
    """Tests for the WaterSample Pydantic model."""

    def test_valid_sample_passes(self):
        """Assert a valid sample passes validation."""
        sample = WaterSample(
            ph=7.0, Hardness=200.0, Solids=20000.0,
            Chloramines=7.0, Sulfate=330.0, Conductivity=400.0,
            Organic_carbon=14.0, Trihalomethanes=60.0, Turbidity=3.5,
        )
        assert sample.ph == 7.0
        assert sample.Hardness == 200.0

    def test_invalid_ph_above_14_raises(self):
        """Assert pH > 14 raises ValidationError (422 in API)."""
        with pytest.raises(ValidationError) as exc_info:
            WaterSample(
                ph=15.0, Hardness=200.0, Solids=20000.0,
                Chloramines=7.0, Sulfate=330.0, Conductivity=400.0,
                Organic_carbon=14.0, Trihalomethanes=60.0, Turbidity=3.5,
            )
        assert "ph" in str(exc_info.value)

    def test_invalid_ph_below_0_raises(self):
        """Assert pH < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            WaterSample(
                ph=-1.0, Hardness=200.0, Solids=20000.0,
                Chloramines=7.0, Sulfate=330.0, Conductivity=400.0,
                Organic_carbon=14.0, Trihalomethanes=60.0, Turbidity=3.5,
            )

    def test_negative_hardness_raises(self):
        """Assert negative Hardness raises ValidationError."""
        with pytest.raises(ValidationError):
            WaterSample(
                ph=7.0, Hardness=-10.0, Solids=20000.0,
                Chloramines=7.0, Sulfate=330.0, Conductivity=400.0,
                Organic_carbon=14.0, Trihalomethanes=60.0, Turbidity=3.5,
            )

    def test_missing_field_raises(self):
        """Assert missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            WaterSample(
                ph=7.0,  # Missing all other fields
            )

    def test_zero_turbidity_raises(self):
        """Assert Turbidity=0 raises (must be >0)."""
        with pytest.raises(ValidationError):
            WaterSample(
                ph=7.0, Hardness=200.0, Solids=20000.0,
                Chloramines=7.0, Sulfate=330.0, Conductivity=400.0,
                Organic_carbon=14.0, Trihalomethanes=60.0, Turbidity=0.0,
            )
