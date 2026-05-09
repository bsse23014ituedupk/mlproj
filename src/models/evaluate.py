"""
Evaluation Module — full metrics suite, threshold optimisation, calibration
check, SHAP explainability, feature importance plots, and overfitting report.

Key additions over the previous version
-----------------------------------------
1. Threshold optimisation   : sweeps 0–1 on the VAL set, picks the cut-point
                              that maximises F1-macro. That threshold is then
                              applied to the TEST set — no leakage.
2. Probability calibration  : Brier score gate triggers isotonic calibration.
3. Feature importance plot  : MDI bar chart saved alongside SHAP plots.
4. Overfitting report       : train/val/test AUC gap analysis with verdicts.
5. MCC tracking             : Matthews Correlation Coefficient reported for
                              all splits, not just test.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    CALIBRATION_CV,
    CALIBRATION_METHOD,
    FEATURE_COLUMNS,
    HEALTH_SIGNIFICANCE,
    OVERFIT_AUC_GAP_THRESHOLD,
    OVERFIT_CV_GAP_THRESHOLD,
    RANDOM_SEED,
    REPORTS_DIR,
    SHAP_MAX_DISPLAY,
    THRESHOLD_SWEEP_STEPS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Brier score above this triggers probability calibration
_BRIER_CALIBRATION_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model_name: str,
    feature_names: list[str],
    cv_roc_auc: float = 0.0,
) -> dict:
    """Run the full evaluation suite on a trained model.

    Steps
    -----
    1. Raw probabilities from the model.
    2. Optional isotonic calibration if Brier score is high.
    3. Threshold sweep on VAL set → optimal cut-point.
    4. Apply optimal threshold to TEST predictions.
    5. Compute all metrics (ROC-AUC, PR-AUC, F1, MCC, Brier, Kappa).
    6. Generate all plots (confusion matrix, ROC, PR, calibration, feature
       importance, learning curve, SHAP).
    7. Print overfitting analysis.

    Args:
        model          : Trained sklearn estimator (post-SMOTE pipeline or raw model).
        X_train/y_train: Training data (used for train-AUC and learning curve).
        X_val/y_val    : Validation data (threshold optimisation only).
        X_test/y_test  : Held-out test data.
        model_name     : Label for logs and plot titles.
        feature_names  : Column names matching the feature matrices.
        cv_roc_auc     : CV ROC-AUC from Optuna (for overfitting gap analysis).

    Returns:
        Dict with all scalar metrics, artifact paths, and calibration flag.
    """

    # ------------------------------------------------------------------
    # 1. Raw probabilities
    # ------------------------------------------------------------------
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val   = model.predict_proba(X_val)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_prob_train)
    val_auc   = roc_auc_score(y_val,   y_prob_val)
    raw_brier = brier_score_loss(y_test, y_prob_test)

    # ------------------------------------------------------------------
    # 2. Probability calibration (isotonic regression) if Brier > gate
    # ------------------------------------------------------------------
    calibrated = False
    if raw_brier > _BRIER_CALIBRATION_THRESHOLD:
        logger.info(
            f"Brier={raw_brier:.4f} > {_BRIER_CALIBRATION_THRESHOLD} — "
            f"applying {CALIBRATION_METHOD} calibration."
        )
        calib_model = CalibratedClassifierCV(
            model, method=CALIBRATION_METHOD, cv=CALIBRATION_CV
        )
        calib_model.fit(X_train, y_train)
        y_prob_val  = calib_model.predict_proba(X_val)[:, 1]
        y_prob_test = calib_model.predict_proba(X_test)[:, 1]
        calibrated  = True
        logger.info("Calibration applied — probabilities re-computed.")
    else:
        logger.info(f"Brier={raw_brier:.4f} ≤ threshold — calibration skipped.")

    # ------------------------------------------------------------------
    # 3. Threshold optimisation on VALIDATION set (never test!)
    # ------------------------------------------------------------------
    best_threshold, val_f1_at_best = _optimise_threshold(y_val, y_prob_val)
    logger.info(
        f"Optimal threshold (val F1-macro={val_f1_at_best:.4f}): {best_threshold:.2f}"
    )

    # ------------------------------------------------------------------
    # 4. Apply threshold to test predictions
    # ------------------------------------------------------------------
    y_pred_test  = (y_prob_test >= best_threshold).astype(int)
    y_pred_train = (y_prob_train >= best_threshold).astype(int)

    # ------------------------------------------------------------------
    # 5. Compute all metrics
    # ------------------------------------------------------------------
    test_auc   = roc_auc_score(y_test, y_prob_test)
    metrics = {
        # AUC across all splits (key overfitting signal)
        "train_roc_auc"    : train_auc,
        "val_roc_auc"      : val_auc,
        "test_roc_auc"     : test_auc,
        "cv_roc_auc"       : cv_roc_auc,
        # Threshold
        "optimal_threshold": round(best_threshold, 4),
        "val_f1_at_threshold": round(val_f1_at_best, 4),
        # PR-AUC (preferred for imbalanced data)
        "test_pr_auc"      : average_precision_score(y_test, y_prob_test),
        # F1
        "test_f1_macro"    : f1_score(y_test, y_pred_test, average="macro"),
        "test_f1_weighted" : f1_score(y_test, y_pred_test, average="weighted"),
        # Precision / Recall (macro)
        "test_precision"   : precision_score(y_test, y_pred_test, average="macro", zero_division=0),
        "test_recall"      : recall_score(y_test, y_pred_test, average="macro"),
        # MCC — the single most robust metric for imbalanced binary classification
        "test_mcc"         : matthews_corrcoef(y_test, y_pred_test),
        "train_mcc"        : matthews_corrcoef(y_train, y_pred_train),
        # Brier score (lower = better probability calibration)
        "test_brier_score" : brier_score_loss(y_test, y_prob_test),
        # Cohen's Kappa — agreement beyond chance
        "test_cohens_kappa": cohen_kappa_score(y_test, y_pred_test),
        # Calibration flag
        "calibrated"       : calibrated,
        "feature_eng_delta_auc": None,
    }

    logger.info(
        f"\n{model_name} — Classification Report (threshold={best_threshold:.2f}):\n"
        f"{classification_report(y_test, y_pred_test, target_names=['Unsafe','Safe'])}"
    )
    logger.info(
        f"{model_name} → AUC={test_auc:.4f} | F1={metrics['test_f1_macro']:.4f} "
        f"| MCC={metrics['test_mcc']:.4f} | Brier={metrics['test_brier_score']:.4f}"
    )

    # ------------------------------------------------------------------
    # 6. Overfitting analysis (logged + saved)
    # ------------------------------------------------------------------
    overfit_report = _overfitting_analysis(
        train_auc, val_auc, test_auc, cv_roc_auc, model_name
    )
    metrics["overfit_report"] = overfit_report

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    artifacts: list[Path] = []
    try:
        artifacts.append(_plot_confusion_matrix(y_test, y_pred_test, model_name, best_threshold))
        artifacts.append(_plot_roc(y_test, y_prob_test, model_name, test_auc))
        artifacts.append(_plot_pr(y_test, y_prob_test, model_name))
        artifacts.append(_plot_calibration(y_test, y_prob_test, model_name))
        artifacts.append(_plot_threshold_sweep(y_val, y_prob_val, model_name, best_threshold))
        # Feature importance (MDI) — only for tree-based models
        try:
            rf = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if hasattr(rf, "feature_importances_"):
                artifacts.append(_plot_feature_importance(rf, feature_names, model_name))
        except Exception as fe:
            logger.warning(f"Feature importance plot skipped: {fe}")
        # Learning curve
        artifacts.append(_plot_learning_curve(model, X_train, y_train, model_name))
    except Exception as e:
        logger.warning(f"Plot generation error: {e}")

    # SHAP
    shap_vals = None
    try:
        rf_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
        shap_vals = _shap_plots(rf_model, X_test, feature_names, model_name)
        if shap_vals is not None:
            artifacts += [REPORTS_DIR / "shap_summary.png", REPORTS_DIR / "shap_bar.png"]
            artifacts.append(_dangerous_factors_report(shap_vals, feature_names))
    except Exception as e:
        logger.warning(f"SHAP error: {e}")

    metrics["artifacts"] = [p for p in artifacts if p and Path(p).exists()]
    metrics["shap_values"] = shap_vals
    return metrics


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def _optimise_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds 0→1 on *validation* data, return argmax of F1-macro.

    Using the validation set (not test) avoids threshold overfitting.

    Args:
        y_true: True labels (validation set).
        y_prob: Predicted probabilities (validation set).

    Returns:
        (best_threshold, best_f1_macro)
    """
    thresholds = np.linspace(0.0, 1.0, THRESHOLD_SWEEP_STEPS)
    best_thr, best_f1 = 0.5, 0.0

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        # Skip degenerate predictions (all one class)
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return float(best_thr), float(best_f1)


# ---------------------------------------------------------------------------
# Overfitting analysis
# ---------------------------------------------------------------------------

def _overfitting_analysis(
    train_auc: float, val_auc: float, test_auc: float,
    cv_auc: float, model_name: str
) -> dict:
    """Compute and log overfitting gap metrics with plain-English verdicts.

    Args:
        train_auc : ROC-AUC on full training set.
        val_auc   : ROC-AUC on validation set.
        test_auc  : ROC-AUC on held-out test set.
        cv_auc    : Mean CV ROC-AUC from Optuna search.
        model_name: For logging.

    Returns:
        Dict with gap metrics and verdict strings.
    """
    train_test_gap = train_auc - test_auc
    cv_test_gap    = cv_auc   - test_auc

    verdict_tt = (
        "⚠️  OVERFITTING"  if train_test_gap > OVERFIT_AUC_GAP_THRESHOLD
        else "✅ ACCEPTABLE"
    )
    verdict_cv = (
        "⚠️  OVERFIT (CV–Test gap)"  if cv_test_gap > OVERFIT_CV_GAP_THRESHOLD
        else "✅ GENERALISING"
    )

    report = {
        "train_auc"       : round(train_auc,       4),
        "val_auc"         : round(val_auc,          4),
        "test_auc"        : round(test_auc,         4),
        "cv_auc"          : round(cv_auc,           4),
        "train_test_gap"  : round(train_test_gap,   4),
        "cv_test_gap"     : round(cv_test_gap,      4),
        "verdict_train_test": verdict_tt,
        "verdict_cv_test"   : verdict_cv,
        "recommendations" : _overfit_recommendations(train_test_gap, cv_test_gap),
    }

    logger.info("\n" + "=" * 60)
    logger.info(f"OVERFITTING ANALYSIS — {model_name}")
    logger.info(f"  Train AUC : {train_auc:.4f}")
    logger.info(f"  Val   AUC : {val_auc:.4f}")
    logger.info(f"  CV    AUC : {cv_auc:.4f}")
    logger.info(f"  Test  AUC : {test_auc:.4f}")
    logger.info(f"  Train–Test gap : {train_test_gap:.4f}  → {verdict_tt}")
    logger.info(f"  CV–Test   gap  : {cv_test_gap:.4f}  → {verdict_cv}")
    for rec in report["recommendations"]:
        logger.info(f"  💡 {rec}")
    logger.info("=" * 60)

    # Persist to disk
    report_path = REPORTS_DIR / f"overfit_analysis_{model_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _overfit_recommendations(train_test_gap: float, cv_test_gap: float) -> list[str]:
    """Return actionable recommendations based on observed gaps."""
    recs = []
    if train_test_gap > OVERFIT_AUC_GAP_THRESHOLD:
        recs += [
            "Increase min_samples_leaf (>= 5) to reduce tree depth.",
            "Lower max_depth (try 8–12) to prevent memorisation.",
            "Increase min_samples_split to require more evidence per split.",
            "Use max_features='sqrt' to decorrelate trees.",
        ]
    if cv_test_gap > OVERFIT_CV_GAP_THRESHOLD:
        recs += [
            "CV–test gap suggests distribution shift; collect more diverse samples.",
            "Consider stricter regularisation (class_weight='balanced_subsample').",
        ]
    if not recs:
        recs.append("No significant overfitting detected — model generalises well.")
    return recs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, name: str, threshold: float
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"],
        xlabel="Predicted", ylabel="True",
        title=f"Confusion Matrix — {name}\n(threshold={threshold:.2f})"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    path = REPORTS_DIR / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, name: str, auc: float) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC Curve — {name}")
    ax.legend(loc="lower right"); plt.tight_layout()
    path = REPORTS_DIR / f"roc_curve_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Path:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall — {name}")
    ax.legend(); plt.tight_layout()
    path = REPORTS_DIR / f"pr_curve_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Path:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, fraction_pos, "s-", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction Positives",
           title=f"Calibration Curve — {name}")
    ax.legend(); plt.tight_layout()
    path = REPORTS_DIR / f"calibration_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_threshold_sweep(
    y_val: np.ndarray, y_prob_val: np.ndarray, name: str, best_thr: float
) -> Path:
    """Plot F1-macro vs threshold so the optimal cut-point is visible."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = []
    for thr in thresholds:
        p = (y_prob_val >= thr).astype(int)
        f1s.append(
            f1_score(y_val, p, average="macro", zero_division=0)
            if len(np.unique(p)) > 1 else 0.0
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1s, lw=2, label="F1-macro (val)")
    ax.axvline(best_thr, color="red", linestyle="--", label=f"Best = {best_thr:.2f}")
    ax.set(xlabel="Threshold", ylabel="F1-macro", title=f"Threshold Sweep — {name}")
    ax.legend(); plt.tight_layout()
    path = REPORTS_DIR / f"threshold_sweep_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_feature_importance(model, feature_names: list[str], name: str) -> Path:
    """MDI (Mean Decrease in Impurity) feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[indices[:top_n]], align="center", color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices[:top_n]], rotation=45, ha="right")
    ax.set(title=f"Feature Importance (MDI) — {name}", ylabel="Importance")
    plt.tight_layout()
    path = REPORTS_DIR / f"feature_importance_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def _plot_learning_curve(model, X_train: np.ndarray, y_train: np.ndarray, name: str) -> Path:
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(1), "o-", label="Train AUC")
    ax.plot(train_sizes, val_scores.mean(1),   "s-", label="CV Val AUC")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15)
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15)
    ax.set(xlabel="Training samples", ylabel="ROC-AUC",
           title=f"Learning Curve — {name}")
    ax.legend(); plt.tight_layout()
    path = REPORTS_DIR / f"learning_curve_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def _shap_plots(
    model, X_test: np.ndarray, feature_names: list[str], name: str
) -> np.ndarray | None:
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — SHAP plots skipped.")
        return None

    X_sample = X_test[:min(200, len(X_test))]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    sv = sv[1] if isinstance(sv, list) else sv

    # Beeswarm
    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title(f"SHAP Summary — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
    plt.title(f"Feature Importance (SHAP) — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plots saved.")
    return sv


def _dangerous_factors_report(shap_values: np.ndarray, feature_names: list[str]) -> Path:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked   = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    lines = [
        "# Dangerous Chemical Factors Report\n",
        "Ranked by SHAP mean |value| — higher = more influential.\n",
        "| Rank | Feature | Mean |SHAP| | Health Significance |",
        "|------|---------|-------------|---------------------|",
    ]
    for rank, (feat, imp) in enumerate(ranked, 1):
        sig = HEALTH_SIGNIFICANCE.get(feat, HEALTH_SIGNIFICANCE.get(feat.split("_")[0], "Engineered feature"))
        lines.append(f"| {rank} | {feat} | {imp:.4f} | {sig} |")
    path = REPORTS_DIR / "dangerous_factors.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Single-sample SHAP for API predictions (unchanged contract)
# ---------------------------------------------------------------------------

def compute_single_shap(model, X_sample: np.ndarray, feature_names: list[str]) -> list[str]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        sv = (sv[1] if isinstance(sv, list) else sv).flatten()
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        return [
            f"{feature_names[i]} ({'↑ increases' if sv[i] > 0 else '↓ decreases'} risk)"
            for i in top_idx
        ]
    except Exception as e:
        get_logger(__name__).warning(f"Single SHAP failed: {e}")
        return ["SHAP unavailable"]
