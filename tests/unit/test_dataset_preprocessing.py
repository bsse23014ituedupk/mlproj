"""
Unit Tests — Dataset preprocessing on real water_potability.csv.

Tests:
- Potability (target) column is correctly dropped from feature matrix X
- IterativeImputer removes all nulls from the real dataset
- Train/val/test splits have correct shapes and no overlap
- Preprocessing is fit only on train (no data leakage)
- Feature count after engineering is exactly 12
- Class balance is logged (61/39 ratio check)
"""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED, RAW_CSV_PATH,
)
from src.data.ingestion import load_raw_data
from src.data.preprocessing import build_preprocessing_pipeline, split_data
from src.features.feature_engineering import (
    create_engineered_features, get_all_feature_names,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load the actual water_potability.csv once for the whole module."""
    if not RAW_CSV_PATH.exists():
        pytest.skip("water_potability.csv not found — skipping real-data tests")
    return load_raw_data()


@pytest.fixture(scope="module")
def engineered_df(raw_df):
    """Apply feature engineering to the raw DataFrame."""
    return create_engineered_features(raw_df)


@pytest.fixture(scope="module")
def split_dfs(engineered_df):
    """Return (train_df, val_df, test_df) splits."""
    return split_data(engineered_df)


# ── Tests: Column Dropping / Target Isolation ────────────────────────────────

class TestTargetIsolation:
    """Verify Potability is never in the feature set."""

    def test_target_not_in_feature_columns(self):
        """Potability must not appear in FEATURE_COLUMNS config list."""
        assert TARGET_COLUMN not in FEATURE_COLUMNS, (
            f"DATA LEAK: '{TARGET_COLUMN}' is in FEATURE_COLUMNS"
        )

    def test_target_not_in_engineered_feature_names(self):
        """Potability must not appear in the full feature name list."""
        all_features = get_all_feature_names()
        assert TARGET_COLUMN not in all_features, (
            f"DATA LEAK: '{TARGET_COLUMN}' found in engineered feature names"
        )

    def test_X_excludes_potability(self, split_dfs):
        """X_train values must not include the Potability column."""
        train_df, _, _ = split_dfs
        all_features = get_all_feature_names()
        # Ensure we can extract X without the target
        assert TARGET_COLUMN not in all_features
        X = train_df[all_features].values
        # Shape: (n_samples, 12) — not 13
        assert X.shape[1] == 12, (
            f"Expected 12 features, got {X.shape[1]}. "
            "Potability may have leaked into X."
        )


# ── Tests: Null Imputation ────────────────────────────────────────────────────

class TestNullImputation:
    """Verify IterativeImputer removes all nulls from the real dataset."""

    def test_raw_data_has_known_nulls(self, raw_df):
        """Confirm the raw dataset has nulls in ph, Sulfate, Trihalomethanes."""
        nulls = raw_df.isnull().sum()
        assert nulls["ph"] > 0,             "Expected nulls in 'ph'"
        assert nulls["Sulfate"] > 0,        "Expected nulls in 'Sulfate'"
        assert nulls["Trihalomethanes"] > 0, "Expected nulls in 'Trihalomethanes'"

    def test_no_nulls_after_preprocessing(self, split_dfs):
        """After pipeline.fit_transform, X_train must have zero nulls."""
        train_df, _, _ = split_dfs
        all_features = get_all_feature_names()
        X_train = train_df[all_features].values

        pipeline = build_preprocessing_pipeline()
        X_processed = pipeline.fit_transform(X_train)

        null_count = int(np.isnan(X_processed).sum())
        assert null_count == 0, (
            f"IterativeImputer left {null_count} nulls in X_train"
        )

    def test_no_nulls_in_val_after_transform(self, split_dfs):
        """Validation set must also have zero nulls after pipeline.transform."""
        train_df, val_df, _ = split_dfs
        all_features = get_all_feature_names()

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(train_df[all_features].values)

        X_val_processed = pipeline.transform(val_df[all_features].values)
        null_count = int(np.isnan(X_val_processed).sum())
        assert null_count == 0, (
            f"Nulls remain in validation set after transform: {null_count}"
        )

    def test_no_nulls_in_test_after_transform(self, split_dfs):
        """Test set must also have zero nulls after pipeline.transform."""
        train_df, _, test_df = split_dfs
        all_features = get_all_feature_names()

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(train_df[all_features].values)

        X_test_processed = pipeline.transform(test_df[all_features].values)
        null_count = int(np.isnan(X_test_processed).sum())
        assert null_count == 0, (
            f"Nulls remain in test set after transform: {null_count}"
        )


# ── Tests: Split Shapes and No Row Overlap ────────────────────────────────────

class TestSplitIntegrity:
    """Verify train/val/test splits are correct and non-overlapping."""

    def test_split_sizes_sum_to_total(self, engineered_df, split_dfs):
        """Train + val + test rows should equal the total dataset rows."""
        train_df, val_df, test_df = split_dfs
        total = len(engineered_df)
        split_sum = len(train_df) + len(val_df) + len(test_df)
        assert split_sum == total, (
            f"Split sum {split_sum} != total {total}"
        )

    def test_test_split_is_20_percent(self, engineered_df, split_dfs):
        """Test set should be approximately 20% of total data."""
        _, _, test_df = split_dfs
        ratio = len(test_df) / len(engineered_df)
        assert 0.18 <= ratio <= 0.22, (
            f"Test split ratio {ratio:.2%} is outside 18–22% range"
        )

    def test_no_row_overlap_train_test(self, split_dfs):
        """Train and test sets must share no rows (by index)."""
        train_df, _, test_df = split_dfs
        overlap = set(train_df.index) & set(test_df.index)
        assert len(overlap) == 0, (
            f"Data leakage: {len(overlap)} rows shared between train and test"
        )

    def test_no_row_overlap_train_val(self, split_dfs):
        """Train and validation sets must share no rows."""
        train_df, val_df, _ = split_dfs
        overlap = set(train_df.index) & set(val_df.index)
        assert len(overlap) == 0, (
            f"Data leakage: {len(overlap)} rows shared between train and val"
        )

    def test_feature_count_is_12(self, split_dfs):
        """After feature engineering, feature count must be exactly 12."""
        train_df, _, _ = split_dfs
        all_features = get_all_feature_names()
        assert len(all_features) == 12, (
            f"Expected 12 features, got {len(all_features)}"
        )
        # All feature columns must exist in train_df
        missing = [f for f in all_features if f not in train_df.columns]
        assert not missing, f"Missing feature columns in train_df: {missing}"


# ── Tests: Class Balance ──────────────────────────────────────────────────────

class TestClassBalance:
    """Verify class distribution in the raw dataset."""

    def test_class_imbalance_is_known(self, raw_df):
        """Class 0 (Unsafe) should outnumber Class 1 (Safe) ~61/39."""
        counts = raw_df[TARGET_COLUMN].value_counts()
        class_0 = counts.get(0, 0)
        class_1 = counts.get(1, 0)
        total = class_0 + class_1
        ratio_0 = class_0 / total
        # Allow +/-5% tolerance around 61%
        assert 0.55 <= ratio_0 <= 0.67, (
            f"Class 0 ratio {ratio_0:.2%} outside expected 55–67% range. "
            f"Counts: class_0={class_0}, class_1={class_1}"
        )

    def test_both_classes_present_in_all_splits(self, split_dfs):
        """All three splits must contain both class 0 and class 1."""
        for split_name, split_df in zip(
            ["train", "val", "test"], split_dfs
        ):
            classes = set(split_df[TARGET_COLUMN].unique())
            assert {0, 1}.issubset(classes), (
                f"{split_name} split is missing a class: {classes}"
            )


# ── Tests: No Infinite Values ────────────────────────────────────────────────

class TestNoInfiniteValues:
    """Verify no infinite values appear after preprocessing."""

    def test_no_inf_after_preprocessing(self, split_dfs):
        """Processed arrays must contain no inf values."""
        train_df, val_df, test_df = split_dfs
        all_features = get_all_feature_names()

        pipeline = build_preprocessing_pipeline()
        X_train = pipeline.fit_transform(train_df[all_features].values)
        X_val   = pipeline.transform(val_df[all_features].values)
        X_test  = pipeline.transform(test_df[all_features].values)

        for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
            assert not np.any(np.isinf(X)), (
                f"Infinite values found in {name} set after preprocessing"
            )
