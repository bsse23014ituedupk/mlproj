"""
Feature Engineering Module — domain-informed derived features.

Creates three chemically-justified interaction features that capture
nonlinear relationships between water quality indicators.
"""
import numpy as np
import pandas as pd
from src.config import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Epsilon to prevent division by zero
_EPS = 1e-6

# Names of the engineered features
ENGINEERED_FEATURES: list[str] = [
    "ph_hardness_ratio",
    "tthm_chloramine_interaction",
    "organic_solids_ratio",
]


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed derived features to the DataFrame.

    Features created:
    1. ph_hardness_ratio = ph / (Hardness + eps)
       Justification: Hard water often neutralizes acidity; this ratio
       captures joint chemical load.

    2. tthm_chloramine_interaction = Trihalomethanes * Chloramines
       Justification: THMs are disinfection byproducts of chloramine
       reactions; their product models synergistic contamination.

    3. organic_solids_ratio = Organic_carbon / (Solids + eps)
       Justification: High organic carbon relative to total dissolved
       solids suggests biological contamination.

    Args:
        df: DataFrame with the 9 original features.

    Returns:
        DataFrame with 3 additional engineered feature columns.
    """
    df = df.copy()

    # Feature 1: pH-Hardness ratio
    # "Hard water often neutralizes acidity; this ratio captures joint chemical load."
    df["ph_hardness_ratio"] = df["ph"] / (df["Hardness"] + _EPS)

    # Feature 2: THM-Chloramine interaction
    # "THMs are disinfection byproducts of chloramine reactions;
    #  their product models synergistic contamination."
    df["tthm_chloramine_interaction"] = df["Trihalomethanes"] * df["Chloramines"]

    # Feature 3: Organic-Solids ratio
    # "High organic carbon relative to total dissolved solids suggests
    #  biological contamination."
    df["organic_solids_ratio"] = df["Organic_carbon"] / (df["Solids"] + _EPS)

    logger.info(f"Created {len(ENGINEERED_FEATURES)} engineered features")
    return df


def create_engineered_features_array(X: np.ndarray) -> np.ndarray:
    """Create engineered features from a numpy array (for pipeline use).

    Assumes columns are in FEATURE_COLUMNS order:
    [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
     Organic_carbon, Trihalomethanes, Turbidity]

    Args:
        X: Feature matrix with shape (n_samples, 9).

    Returns:
        Feature matrix with shape (n_samples, 12) — 9 original + 3 new.
    """
    # Column indices (matching FEATURE_COLUMNS order)
    ph_idx = 0
    hardness_idx = 1
    solids_idx = 2
    chloramines_idx = 3
    organic_carbon_idx = 6
    tthm_idx = 7

    ph_hardness = X[:, ph_idx] / (X[:, hardness_idx] + _EPS)
    tthm_chlor = X[:, tthm_idx] * X[:, chloramines_idx]
    org_solids = X[:, organic_carbon_idx] / (X[:, solids_idx] + _EPS)

    engineered = np.column_stack([X, ph_hardness, tthm_chlor, org_solids])
    return engineered


def get_all_feature_names() -> list[str]:
    """Return the full list of feature names including engineered ones.

    Returns:
        List of 12 feature names (9 original + 3 engineered).
    """
    return FEATURE_COLUMNS + ENGINEERED_FEATURES
