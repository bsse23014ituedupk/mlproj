"""
Preprocessing Module — builds a scikit-learn Pipeline for data transformation.

Pipeline steps (in order):
1. IterativeImputer (MICE) — for ph, Sulfate, Trihalomethanes
2. IQR Outlier Clipping — custom transformer
3. RobustScaler — median/IQR based scaling
4. SMOTE — applied only on training data (handled externally)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE, IQR_CLIP_FACTOR, SMOTE_K_NEIGHBORS,
    SPLITS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """Custom transformer that clips outliers at 1.5×IQR boundaries.

    Justification: Solids and Hardness have extreme tails; clipping at
    1.5×IQR before scaling prevents distortion of distance-based models
    (KNN) and improves convergence of Logistic Regression.
    """

    def __init__(self, factor: float = 1.5):
        """Initialize the clipper.

        Args:
            factor: IQR multiplier for defining outlier bounds.
        """
        self.factor = factor
        self.lower_bounds_: np.ndarray = None
        self.upper_bounds_: np.ndarray = None

    def fit(self, X: np.ndarray, y=None) -> "IQROutlierClipper":
        """Compute IQR bounds from training data only.

        Args:
            X: Training feature matrix.
            y: Ignored (API compatibility).

        Returns:
            self
        """
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to the learned IQR bounds.

        Args:
            X: Feature matrix to clip.

        Returns:
            Clipped feature matrix.
        """
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped


def build_preprocessing_pipeline() -> Pipeline:
    """Build the preprocessing pipeline with named steps.

    Returns:
        A scikit-learn Pipeline with imputer, clipper, and scaler.

    Note:
        SMOTE is NOT part of this preprocessing pipeline — it is handled
        within the imblearn Pipeline during cross-validation to prevent
        information leakage.
    """
    pipeline = Pipeline([
        # Step 1: MICE imputation
        # "KNN and median imputation ignore feature correlations; MICE models
        # each feature as a function of others, producing less biased estimates
        # on MAR (Missing At Random) data."
        ("imputer", IterativeImputer(
            max_iter=20,
            random_state=RANDOM_SEED,
            sample_posterior=False,
        )),

        # Step 2: IQR outlier clipping
        # "Solids and Hardness have extreme tails; clipping at 1.5×IQR before
        # scaling prevents distortion of distance-based models."
        ("outlier_clipper", IQROutlierClipper(factor=IQR_CLIP_FACTOR)),

        # Step 3: RobustScaler
        # "RobustScaler uses median/IQR and is resistant to the outliers
        # confirmed in EDA. StandardScaler would be skewed by extreme values."
        ("scaler", RobustScaler()),
    ])

    logger.info("Built preprocessing pipeline: imputer → clipper → scaler")
    return pipeline


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets with stratification.

    Args:
        df: Full dataset with features and target.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Use all columns except the target — preserves engineered features
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Second split: separate validation set (10% of original = 12.5% of remaining)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_temp
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save splits to disk for reproducibility
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    logger.info(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance class distribution.

    Justification: The 61/39 class split biases most classifiers toward
    the majority class (Unsafe). SMOTE synthesizes minority-class samples
    in feature space — not by simple duplication — without information
    leakage because it is called ONLY on training data.

    Args:
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Tuple of (X_resampled, y_resampled) with balanced classes.
    """
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        f"SMOTE applied — original: {len(y)} samples, "
        f"resampled: {len(y_res)} samples"
    )
    return X_res, y_res
