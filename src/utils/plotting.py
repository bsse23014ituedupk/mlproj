"""
Reusable visualization helpers for the Water Potability Classifier.

All functions save figures to the ``reports/`` directory and return the
``matplotlib.figure.Figure`` object for optional further customization.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, KFold

from src.config import REPORTS_DIR, RANDOM_SEED
from src.utils.logger import get_logger

# Use non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")

logger = get_logger(__name__)

# Consistent style across all plots
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the reports directory.

    Args:
        fig: The matplotlib Figure to save.
        filename: Filename (e.g. ``confusion_matrix.png``).

    Returns:
        Absolute path to the saved figure.
    """
    filepath = REPORTS_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved plot: {filepath}")
    return filepath


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Path:
    """Plot confusion matrix with absolute counts and row-normalized percentages.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name of the model (for the title).

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Unsafe", "Safe"],
        cmap="Blues", ax=axes[0],
    )
    axes[0].set_title(f"{model_name} — Counts")

    # Row-normalized (recall per class)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Unsafe", "Safe"],
        normalize="true", cmap="Blues", ax=axes[1], values_format=".2%",
    )
    axes[1].set_title(f"{model_name} — Normalized")

    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "confusion_matrix.png")


def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Path:
    """Plot ROC curve with AUC annotation.

    Args:
        model: Trained classifier with ``predict_proba``.
        X_test: Test feature matrix.
        y_test: Test labels.
        model_name: Name for the title.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)

    # Diagonal baseline (random classifier)
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC=0.50)")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, "roc_curve.png")


def plot_precision_recall_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Path:
    """Plot Precision-Recall curve with Average Precision annotation.

    Args:
        model: Trained classifier with ``predict_proba``.
        X_test: Test feature matrix.
        y_test: Test labels.
        model_name: Name for the title.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "precision_recall_curve.png")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> Path:
    """Plot calibration curve to assess probability output quality.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Name for the title.
        n_bins: Number of bins for calibration.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )

    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, "calibration_curve.png")


def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    cv: int = 5,
) -> Path:
    """Plot learning curve to detect overfitting/underfitting.

    Args:
        model: Untrained or cloned estimator.
        X_train: Training feature matrix.
        y_train: Training labels.
        model_name: Name for the title.
        cv: Number of CV folds.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_SEED,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC-AUC Score")
    ax.set_title(f"Learning Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, "learning_curve.png")


def plot_feature_importance_bar(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
) -> Path:
    """Plot horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance values (e.g., mean |SHAP|).
        model_name: Name for the title.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_idx = np.argsort(importances)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=sns.color_palette("viridis", len(feature_names)),
    )
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Feature Importance (SHAP) — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "shap_bar.png")
