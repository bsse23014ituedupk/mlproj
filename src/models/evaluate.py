"""
Evaluation Module — comprehensive metrics and visualizations on held-out test set.

Produces: ROC-AUC, PR-AUC, F1, MCC, Brier Score, Cohen's Kappa, SHAP plots,
confusion matrix, calibration curve, learning curve, and dangerous factors report.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, brier_score_loss,
    cohen_kappa_score, classification_report,
)
from src.config import (
    REPORTS_DIR, SHAP_MAX_DISPLAY, HEALTH_SIGNIFICANCE,
    FEATURE_COLUMNS, RANDOM_SEED,
)
from src.utils.logger import get_logger
from src.utils.plotting import (
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_calibration_curve,
    plot_learning_curve,
)

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model_name: str,
    feature_names: list[str],
) -> dict:
    """Run full evaluation suite on a trained model.

    Args:
        model: Trained sklearn estimator.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name for logging and plot titles.
        feature_names: List of feature names.

    Returns:
        Dict with all metrics and artifact paths.
    """
    # --- Predictions ----------------------------------------------------------
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]

    # --- Metrics --------------------------------------------------------------
    metrics = {
        "train_roc_auc": roc_auc_score(y_train, y_prob_train),
        "val_roc_auc": roc_auc_score(y_val, y_prob_val),
        "test_roc_auc": roc_auc_score(y_test, y_prob_test),
        # PR-AUC preferred for imbalanced data — focuses on minority class
        "test_pr_auc": average_precision_score(y_test, y_prob_test),
        "test_f1_macro": f1_score(y_test, y_pred_test, average="macro"),
        "test_f1_weighted": f1_score(y_test, y_pred_test, average="weighted"),
        "test_precision": precision_score(y_test, y_pred_test, average="macro"),
        "test_recall": recall_score(y_test, y_pred_test, average="macro"),
        # MCC: "the only metric robust to class imbalance that considers
        # all four confusion matrix quadrants"
        "test_mcc": matthews_corrcoef(y_test, y_pred_test),
        # Brier Score: probability calibration quality (lower is better)
        "test_brier_score": brier_score_loss(y_test, y_prob_test),
        # Cohen's Kappa: agreement beyond chance
        "test_cohens_kappa": cohen_kappa_score(y_test, y_pred_test),
        "feature_eng_delta_auc": None,  # Set later if applicable
    }

    logger.info(f"\n{model_name} — Classification Report:\n"
                f"{classification_report(y_test, y_pred_test, target_names=['Unsafe', 'Safe'])}")
    logger.info(f"{model_name} Metrics: AUC={metrics['test_roc_auc']:.4f}, "
                f"F1={metrics['test_f1_macro']:.4f}, MCC={metrics['test_mcc']:.4f}")

    # --- Visualizations -------------------------------------------------------
    artifacts = []
    try:
        artifacts.append(plot_confusion_matrix(y_test, y_pred_test, model_name))
        artifacts.append(plot_roc_curve(model, X_test, y_test, model_name))
        artifacts.append(plot_precision_recall_curve(model, X_test, y_test, model_name))
        artifacts.append(plot_calibration_curve(y_test, y_prob_test, model_name))
        artifacts.append(plot_learning_curve(model, X_train, y_train, model_name))
    except Exception as e:
        logger.warning(f"Plot generation error: {e}")

    # --- SHAP Explanations ----------------------------------------------------
    shap_values_global = None
    try:
        shap_values_global = _compute_and_plot_shap(
            model, X_test, feature_names, model_name
        )
        if shap_values_global is not None:
            artifacts.append(REPORTS_DIR / "shap_summary.png")
            artifacts.append(REPORTS_DIR / "shap_bar.png")
            # Generate dangerous factors report
            report_path = _generate_dangerous_factors_report(
                shap_values_global, feature_names
            )
            artifacts.append(report_path)
    except Exception as e:
        logger.warning(f"SHAP computation error: {e}")

    metrics["artifacts"] = [p for p in artifacts if p and Path(p).exists()]
    metrics["shap_values"] = shap_values_global
    return metrics


def _compute_and_plot_shap(
    model, X_test: np.ndarray, feature_names: list[str], model_name: str
) -> np.ndarray:
    """Compute SHAP values and generate summary + bar plots.

    SHAP values quantify each feature's marginal contribution to the
    prediction using Shapley values from cooperative game theory.

    Args:
        model: Trained model.
        X_test: Test features.
        feature_names: Feature names for labeling.
        model_name: Model name for titles.

    Returns:
        SHAP values array or None if computation fails.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info(f"Computing SHAP values for {model_name}...")

    # Use a subset for speed if dataset is large
    X_sample = X_test[:min(200, len(X_test))]

    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # For binary classification, take class 1 SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # SHAP Summary (beeswarm) plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample[:len(shap_vals)],
                      feature_names=feature_names, show=False,
                      max_display=SHAP_MAX_DISPLAY)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample[:len(shap_vals)],
                      feature_names=feature_names, plot_type="bar",
                      show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title(f"Feature Importance (SHAP) — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plots saved")
    return shap_vals


def _generate_dangerous_factors_report(
    shap_values: np.ndarray, feature_names: list[str]
) -> Path:
    """Generate a ranked Markdown table of dangerous chemical factors.

    Args:
        shap_values: SHAP values array.
        feature_names: Feature names.

    Returns:
        Path to the saved Markdown report.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Pair features with their importance
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    lines = [
        "# Dangerous Chemical Factors Report\n",
        "Ranked by SHAP mean absolute value (higher = more influential).\n",
        "| Rank | Feature | Mean |SHAP| | Health Significance |",
        "|------|---------|-------------|---------------------|",
    ]

    for rank, (feat, imp) in enumerate(feature_importance, 1):
        # Look up health significance from config
        base_feat = feat.split("_")[0] if feat in HEALTH_SIGNIFICANCE else feat
        significance = HEALTH_SIGNIFICANCE.get(
            feat, HEALTH_SIGNIFICANCE.get(base_feat, "Engineered feature")
        )
        lines.append(f"| {rank} | {feat} | {imp:.4f} | {significance} |")

    report_path = REPORTS_DIR / "dangerous_factors.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Dangerous factors report saved to {report_path}")
    return report_path


def compute_single_shap(
    model, X_sample: np.ndarray, feature_names: list[str]
) -> list[dict]:
    """Compute SHAP values for a single sample (for API predictions).

    Args:
        model: Trained model.
        X_sample: Single sample array of shape (1, n_features).
        feature_names: Feature names.

    Returns:
        List of top 3 risk factors with direction.
    """
    try:
        import shap
        # Use TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()

        # Top 3 by absolute value
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        factors = []
        for idx in top_idx:
            direction = "↑ increases risk" if sv[idx] > 0 else "↓ decreases risk"
            factors.append(f"{feature_names[idx]} ({direction})")
        return factors
    except Exception as e:
        logger.warning(f"Single SHAP computation failed: {e}")
        return ["SHAP unavailable"]
