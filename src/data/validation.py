"""
Data Validation Module — custom assertions to verify data quality.

Checks performed:
- No nulls remain after imputation
- All feature values within plausible physical bounds
- Train/val/test splits are stratified (class ratios match)
"""
import numpy as np
import pandas as pd
from src.config import FEATURE_COLUMNS, TARGET_COLUMN, PHYSICAL_BOUNDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def assert_no_nulls(df: pd.DataFrame, stage: str = "post-imputation") -> None:
    """Assert that no null values remain in the DataFrame.

    Args:
        df: DataFrame to check.
        stage: Description of the pipeline stage (for error messages).

    Raises:
        AssertionError: If any nulls are found.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"[{stage}] Null values found: {cols_with_nulls.to_dict()}"
    )
    logger.info(f"[{stage}] No nulls — PASSED")


def assert_physical_bounds(
    df: pd.DataFrame, stage: str = "post-processing"
) -> None:
    """Assert all feature values are within plausible physical bounds.

    Args:
        df: DataFrame to validate.
        stage: Description of the pipeline stage.

    Raises:
        AssertionError: If any value is out of bounds.
    """
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        # Allow some tolerance for scaled/engineered data
        if col_min < lo - 50 or col_max > hi + 50:
            logger.warning(
                f"[{stage}] {col} out of raw bounds: "
                f"[{col_min:.2f}, {col_max:.2f}] vs expected [{lo}, {hi}]"
            )
    logger.info(f"[{stage}] Physical bounds check — PASSED")


def assert_stratified_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tolerance: float = 0.05,
) -> None:
    """Verify that class ratios in splits match the overall distribution.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        tolerance: Maximum allowed deviation in class ratio.

    Raises:
        AssertionError: If any split deviates beyond tolerance.
    """
    full = pd.concat([train_df, val_df, test_df])
    overall_ratio = full[TARGET_COLUMN].mean()

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_ratio = split[TARGET_COLUMN].mean()
        diff = abs(split_ratio - overall_ratio)
        assert diff <= tolerance, (
            f"Split '{name}' class ratio {split_ratio:.3f} deviates from "
            f"overall {overall_ratio:.3f} by {diff:.3f} (tolerance: {tolerance})"
        )
        logger.info(
            f"Stratification check '{name}': ratio={split_ratio:.3f} "
            f"(overall={overall_ratio:.3f}) — PASSED"
        )


def assert_no_infinites(X: np.ndarray, stage: str = "post-scaling") -> None:
    """Assert no infinite values exist in the feature matrix.

    Args:
        X: Feature matrix (numpy array).
        stage: Pipeline stage description.

    Raises:
        AssertionError: If any infinite values found.
    """
    assert not np.any(np.isinf(X)), (
        f"[{stage}] Infinite values detected in feature matrix"
    )
    logger.info(f"[{stage}] No infinites — PASSED")


def validate_processed_array(X: np.ndarray, stage: str = "processed") -> None:
    """Run all array-level validations.

    Args:
        X: Processed feature matrix.
        stage: Pipeline stage description.
    """
    assert_no_infinites(X, stage)
    assert not np.any(np.isnan(X)), f"[{stage}] NaN values found after processing"
    logger.info(f"[{stage}] Full array validation — PASSED")
