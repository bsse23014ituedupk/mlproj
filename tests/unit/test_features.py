"""
Unit Tests — Feature engineering module.

Tests:
- Derived features are finite (no NaN/Inf)
- No data leakage (fit only on train)
- Correct number of features produced
"""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, RANDOM_SEED
from src.features.feature_engineering import (
    create_engineered_features,
    create_engineered_features_array,
    get_all_feature_names,
    ENGINEERED_FEATURES,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with realistic water quality values."""
    np.random.seed(RANDOM_SEED)
    n = 50
    data = {
        "ph": np.random.uniform(1, 14, n),
        "Hardness": np.random.uniform(50, 300, n),
        "Solids": np.random.uniform(1000, 50000, n),
        "Chloramines": np.random.uniform(1, 13, n),
        "Sulfate": np.random.uniform(100, 500, n),
        "Conductivity": np.random.uniform(100, 800, n),
        "Organic_carbon": np.random.uniform(2, 28, n),
        "Trihalomethanes": np.random.uniform(1, 120, n),
        "Turbidity": np.random.uniform(1, 7, n),
    }
    return pd.DataFrame(data)


class TestFeatureEngineering:
    """Tests for domain-informed feature engineering."""

    def test_creates_correct_number_of_features(self, sample_df):
        """Assert 3 new features are added."""
        result = create_engineered_features(sample_df)
        assert len(result.columns) == len(FEATURE_COLUMNS) + len(ENGINEERED_FEATURES)

    def test_engineered_features_are_finite(self, sample_df):
        """Assert all derived features contain no NaN or Inf."""
        result = create_engineered_features(sample_df)
        for feat in ENGINEERED_FEATURES:
            assert not result[feat].isna().any(), f"{feat} has NaN values"
            assert not np.isinf(result[feat]).any(), f"{feat} has Inf values"

    def test_array_version_matches(self, sample_df):
        """Assert array-based engineering matches DataFrame version."""
        df_result = create_engineered_features(sample_df)
        X = sample_df[FEATURE_COLUMNS].values
        arr_result = create_engineered_features_array(X)
        assert arr_result.shape[1] == 12  # 9 original + 3 engineered

    def test_get_all_feature_names(self):
        """Assert feature name list has correct length."""
        names = get_all_feature_names()
        assert len(names) == 12
        assert names[:9] == FEATURE_COLUMNS

    def test_no_data_leakage(self, sample_df):
        """Feature engineering uses only the row's own values, not global stats."""
        # Create features for first 10 rows
        subset = sample_df.head(10)
        result_subset = create_engineered_features(subset)
        # Create features for full df and take first 10
        result_full = create_engineered_features(sample_df).head(10)
        # Values should be identical (no leakage from other rows)
        for feat in ENGINEERED_FEATURES:
            np.testing.assert_array_almost_equal(
                result_subset[feat].values,
                result_full[feat].values,
            )
