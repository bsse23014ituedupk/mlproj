"""
Configuration module — Single source of truth for all hyperparameters,
file paths, random seeds, and constants used across the project.

This centralises every 'magic number' so that:
1. Nothing is hard-coded in business logic.
2. Reproducing any experiment requires changing only this file.
3. Viva examiners can inspect every design choice in one place.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Load .env file if present (for local development)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # water_potability_classifier/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure critical directories exist at import time
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Raw CSV path
RAW_CSV_PATH = RAW_DATA_DIR / "water_potability.csv"

# ---------------------------------------------------------------------------
# Random Seed — fixed for full reproducibility across all stochastic steps
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Data Splitting — stratified to preserve the 61/39 class ratio
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20   # 20% held-out test set
VAL_SIZE: float = 0.10    # 10% validation set (from remaining 80%)

# ---------------------------------------------------------------------------
# Feature Columns — the 9 numeric water-quality indicators
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "ph",               # Acidity/alkalinity (WHO safe: 6.5–8.5)
    "Hardness",         # Calcium carbonate concentration (mg/L)
    "Solids",           # Total dissolved solids (ppm)
    "Chloramines",      # Chloramine disinfectant level (ppm)
    "Sulfate",          # Sulfate ion concentration (mg/L)
    "Conductivity",     # Electrical conductivity (μS/cm)
    "Organic_carbon",   # Organic carbon content (ppm)
    "Trihalomethanes",  # THM disinfection byproducts (μg/L)
    "Turbidity",        # Cloudiness / suspended particles (NTU)
]

TARGET_COLUMN: str = "Potability"  # Binary: 0 = Not Safe, 1 = Safe

# Columns known to have missing values in the raw dataset
COLUMNS_WITH_NULLS: list[str] = ["ph", "Sulfate", "Trihalomethanes"]

# ---------------------------------------------------------------------------
# Physical Bounds — used by validation.py to catch impossible values
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "ph":               (0.0, 14.0),      # pH scale
    "Hardness":         (0.0, 1000.0),     # mg/L CaCO3
    "Solids":           (0.0, 100000.0),   # ppm
    "Chloramines":      (0.0, 20.0),       # ppm
    "Sulfate":          (0.0, 1000.0),     # mg/L
    "Conductivity":     (0.0, 1500.0),     # μS/cm
    "Organic_carbon":   (0.0, 30.0),       # ppm
    "Trihalomethanes":  (0.0, 200.0),      # μg/L
    "Turbidity":        (0.0, 10.0),       # NTU
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IQR_CLIP_FACTOR: float = 1.5  # Outlier clipping at 1.5×IQR (standard Tukey fence)
SMOTE_K_NEIGHBORS: int = 5     # Number of nearest neighbours for SMOTE synthesis

# ---------------------------------------------------------------------------
# Model Hyperparameter Search Grids
# ---------------------------------------------------------------------------
RANDOM_FOREST_GRID: dict = {
    "n_estimators": [200, 300, 500],          # More trees → lower variance
    "max_depth": [8, 12, 15, 20, None],       # None allows full depth (with min_samples guards)
    "min_samples_split": [2, 5],              # Tighter set for faster search
    "min_samples_leaf": [1, 2],              # Lower leaf size → finer decision boundaries
    "max_features": ["sqrt", "log2", 0.5],  # 0.5 = half features per split
    "class_weight": ["balanced", "balanced_subsample"],  # Handle class imbalance
}

# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------
CV_N_SPLITS: int = 5          # 5-fold Stratified K-Fold cross-validation
CV_SCORING: str = "roc_auc"   # Primary CV metric — best for ranking safe/unsafe
CV_N_JOBS: int = -1           # Use all CPU cores

# ---------------------------------------------------------------------------
# Optuna Bayesian Hyperparameter Optimisation
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS: int = 80     # Trial budget (covers search space thoroughly)
OPTUNA_TIMEOUT: int = 1800    # Max wall-clock seconds (30 min safety cap)
OPTUNA_DIRECTION: str = "maximize"  # We maximise CV ROC-AUC

# ---------------------------------------------------------------------------
# Probability Calibration
# ---------------------------------------------------------------------------
# CalibratedClassifierCV with 'isotonic' is used when the test Brier score
# exceeds 0.20 — a pragmatic threshold indicating poor probability alignment.
CALIBRATION_METHOD: str = "isotonic"  # 'isotonic' or 'sigmoid'
CALIBRATION_CV: int = 5               # Internal CV folds for calibration

# ---------------------------------------------------------------------------
# Threshold Optimisation
# ---------------------------------------------------------------------------
# Instead of the default 0.5 decision threshold, we sweep the full probability
# range and select the cut-point that maximises F1-macro on the VALIDATION set
# (not the test set — prevents threshold overfitting).
THRESHOLD_SWEEP_STEPS: int = 101  # 0.00, 0.01, ..., 1.00

# ---------------------------------------------------------------------------
# Overfitting Analysis
# ---------------------------------------------------------------------------
# A train/test ROC-AUC gap above this value triggers an overfitting warning.
OVERFIT_AUC_GAP_THRESHOLD: float = 0.10
# A CV/test ROC-AUC gap above this value also flags potential overfit.
OVERFIT_CV_GAP_THRESHOLD: float = 0.08

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "water_potability_v1")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "water_potability_classifier")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", str(PROJECT_ROOT / "training.log"))  # File logging path


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_MAX_DISPLAY: int = 10  # Max features shown in SHAP plots

# ---------------------------------------------------------------------------
# Health significance descriptions for the dangerous factors report
# ---------------------------------------------------------------------------
HEALTH_SIGNIFICANCE: dict[str, str] = {
    "ph":               "WHO safe range 6.5–8.5; extreme pH causes GI irritation",
    "Hardness":         "High hardness (>300 mg/L) linked to cardiovascular risk",
    "Solids":           "TDS >500 ppm affects taste; >1000 ppm is unpalatable",
    "Chloramines":      "EPA limit 4 mg/L; excess causes eye/nose irritation",
    "Sulfate":          "Laxative effect >500 mg/L (EPA secondary standard)",
    "Conductivity":     "Proxy for ionic content; WHO guideline <400 μS/cm",
    "Organic_carbon":   "Indicator of biological contamination; EPA limit 2 mg/L",
    "Trihalomethanes":  "Carcinogenic DBPs; EPA MCL 80 μg/L",
    "Turbidity":        "Must be <4 NTU (WHO); high turbidity harbours pathogens",
}
