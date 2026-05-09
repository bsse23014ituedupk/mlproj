"""
Main Training Pipeline — end-to-end water potability classifier.

Pipeline stages
---------------
1.  Data ingestion           : load raw CSV, log metadata
2.  Feature engineering      : add 3 domain-informed derived features
3.  Stratified split         : train / val / test (70 / 10 / 20 %)
4.  Preprocessing (train-fit only): MICE imputation → IQR clipping → RobustScaler
5.  Baseline                 : DummyClassifier sets a minimum performance bar
6.  Optuna search            : Bayesian hyperparameter tuning on 5-fold StratifiedKFold
                               with SMOTE inside each fold (zero leakage)
7.  Final fit                : best pipeline fit on full training set
8.  Evaluation               : threshold-optimised metrics, calibration check,
                               SHAP explainability, overfitting analysis
9.  MLflow logging           : params, metrics, artefacts, model
10. Persist artefacts        : model.joblib, preprocessor.joblib, metadata.json

Anti-overfitting measures
--------------------------
* SMOTE inside CV folds — synthetic samples never contaminate validation data.
* Optuna search space biased toward regularising parameters (capped max_depth,
  min_samples_leaf ≥ 2, min_samples_split ≥ 5).
* OOB score tracked as a free diagnostic.
* Threshold chosen on VAL set, evaluated on TEST set.
* Overfitting gap report written to reports/ after every run.
"""
from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.data.ingestion import download_dataset, load_raw_data, log_dataset_metadata
from src.data.preprocessing import apply_smote, build_preprocessing_pipeline, split_data
from src.data.validation import (
    assert_no_nulls,
    assert_stratified_splits,
    validate_processed_array,
)
from src.features.feature_engineering import create_engineered_features, get_all_feature_names
from src.models.baseline import run_baseline
from src.models.evaluate import evaluate_model
from src.models.optimize import build_best_pipeline, run_kmeans_analysis, run_optuna_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "RandomForest"

SELECTION_RATIONALE = (
    "Random Forest — bagging ensemble of decision trees. "
    "Reduces variance via bootstrap aggregation and random feature sub-sampling. "
    "Naturally handles mixed-scale features without needing normalization, "
    "making it robust on this 9-feature water quality dataset."
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> dict:
    """Execute the complete training pipeline and return result summary.

    Returns:
        Dict containing best_model name, test metrics, and artefact paths.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # -----------------------------------------------------------------------
    # STEP 1 — Data Ingestion
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    download_dataset()
    df = load_raw_data()
    logger.info(f"Raw dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    logger.info(f"Class distribution:\n{df[TARGET_COLUMN].value_counts().to_dict()}")
    logger.info(
        f"Null counts:\n"
        f"  ph={df['ph'].isnull().sum()}, "
        f"Sulfate={df['Sulfate'].isnull().sum()}, "
        f"Trihalomethanes={df['Trihalomethanes'].isnull().sum()}"
    )

    # -----------------------------------------------------------------------
    # STEP 2 — Feature Engineering (applied before split to avoid index issues)
    # Note: all derived features use only X values — no leakage from y.
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    df = create_engineered_features(df)
    all_features = get_all_feature_names()
    logger.info(f"Feature set ({len(all_features)}): {all_features}")

    # Safety guard — target column must not appear in feature list
    assert TARGET_COLUMN not in all_features, (
        f"DATA LEAK DETECTED: '{TARGET_COLUMN}' found in feature list!"
    )

    # -----------------------------------------------------------------------
    # STEP 3 — Stratified Train / Val / Test Split
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Stratified Data Splitting (70/10/20)")
    train_df, val_df, test_df = split_data(df)
    assert_stratified_splits(train_df, val_df, test_df)
    logger.info(
        f"Split sizes — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )
    logger.info(f"Train class balance: {train_df[TARGET_COLUMN].value_counts().to_dict()}")

    X_train_raw = train_df[all_features].values
    y_train      = train_df[TARGET_COLUMN].values
    X_val_raw   = val_df[all_features].values
    y_val        = val_df[TARGET_COLUMN].values
    X_test_raw  = test_df[all_features].values
    y_test       = test_df[TARGET_COLUMN].values

    # -----------------------------------------------------------------------
    # STEP 4 — Preprocessing (fit on TRAIN only — zero leakage)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocessing (impute → clip → scale)")
    logger.info("  Fitting pipeline on TRAINING set ONLY.")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train_raw)   # fit here
    X_val_proc   = preprocessor.transform(X_val_raw)          # transform only
    X_test_proc  = preprocessor.transform(X_test_raw)         # transform only

    logger.info(
        f"Post-null check — train: {int(np.isnan(X_train_proc).sum())}, "
        f"val: {int(np.isnan(X_val_proc).sum())}, "
        f"test: {int(np.isnan(X_test_proc).sum())} (all must be 0)"
    )
    validate_processed_array(X_train_proc, "train")
    validate_processed_array(X_val_proc,   "val")
    validate_processed_array(X_test_proc,  "test")

    # Persist preprocessor for API use
    joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.joblib")
    logger.info("Preprocessor saved.")

    # -----------------------------------------------------------------------
    # STEP 5a — Baseline (DummyClassifier — must be beaten)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5a: Baseline DummyClassifier")
    baseline_metrics = run_baseline(X_train_proc, y_train, X_test_proc, y_test)
    baseline_auc = baseline_metrics.get("test_roc_auc", 0.5)
    logger.info(f"Baseline AUC floor: {baseline_auc:.4f}")

    # -----------------------------------------------------------------------
    # STEP 5b — K-Means Bonus Analysis (ARI diagnostic)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5b: K-Means Cluster Analysis")
    ari = run_kmeans_analysis(X_train_proc, y_train)

    # -----------------------------------------------------------------------
    # STEP 6 — Optuna Hyperparameter Optimisation
    #
    # SMOTE is embedded inside the imblearn Pipeline used in each Optuna trial.
    # The CV folds are StratifiedKFold — class ratio preserved in every fold.
    # The preprocessed (but NOT SMOTE'd) training data is passed here so that
    # Optuna only tunes model hyperparameters; the scaler is already fitted.
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Optuna Bayesian Hyperparameter Search")
    best_params, best_cv_score, best_cv_std = run_optuna_search(
        MODEL_NAME, X_train_proc, y_train
    )
    logger.info(f"Best CV ROC-AUC: {best_cv_score:.4f} ± {best_cv_std:.4f}")
    logger.info(f"Best params: {best_params}")

    # -----------------------------------------------------------------------
    # STEP 7 — Final Fit on Full Training Set
    #
    # Re-build pipeline with best params and fit on all training data (not
    # just one fold). SMOTE is applied once to the full training set here.
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Final Model Training on Full Training Set")
    final_pipeline = build_best_pipeline(best_params)
    final_pipeline.fit(X_train_proc, y_train)

    best_model = final_pipeline.named_steps["model"]

    # OOB score — free diagnostic that does not touch val/test data
    if hasattr(best_model, "oob_score_"):
        logger.info(f"OOB score (free diagnostic): {best_model.oob_score_:.4f}")

    # -----------------------------------------------------------------------
    # STEP 8 — Comprehensive Evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8: Evaluation (threshold-optimised, calibration-checked)")
    metrics = evaluate_model(
        model=best_model,
        X_train=X_train_proc, y_train=y_train,
        X_val=X_val_proc,     y_val=y_val,
        X_test=X_test_proc,   y_test=y_test,
        model_name=MODEL_NAME,
        feature_names=all_features,
        cv_roc_auc=best_cv_score,
    )

    # Compute improvement over baseline
    delta_auc = metrics["test_roc_auc"] - baseline_auc
    logger.info(f"Improvement over baseline AUC: +{delta_auc:.4f}")

    # -----------------------------------------------------------------------
    # STEP 9 — MLflow Logging
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 9: MLflow Logging")
    with mlflow.start_run(run_name=MODEL_NAME):
        log_dataset_metadata(df)
        mlflow.log_params({
            "model_type"        : MODEL_NAME,
            "n_features"        : len(all_features),
            "smote_applied"     : True,
            "imputer"           : "IterativeImputer(MICE)",
            "scaler"            : "RobustScaler",
            "cv_strategy"       : "StratifiedKFold-5",
            "hp_optimizer"      : "Optuna-TPE",
            "calibrated"        : metrics["calibrated"],
            "optimal_threshold" : metrics["optimal_threshold"],
            **{f"rf_{k}": v for k, v in best_params.items()},
        })
        mlflow.log_metrics({
            "cv_roc_auc"        : best_cv_score,
            "cv_roc_auc_std"    : best_cv_std,
            "train_roc_auc"     : metrics["train_roc_auc"],
            "val_roc_auc"       : metrics["val_roc_auc"],
            "test_roc_auc"      : metrics["test_roc_auc"],
            "test_pr_auc"       : metrics["test_pr_auc"],
            "test_f1_macro"     : metrics["test_f1_macro"],
            "test_precision"    : metrics["test_precision"],
            "test_recall"       : metrics["test_recall"],
            "test_mcc"          : metrics["test_mcc"],
            "test_brier_score"  : metrics["test_brier_score"],
            "test_cohens_kappa" : metrics["test_cohens_kappa"],
            "delta_auc_baseline": delta_auc,
            "kmeans_ari"        : ari,
        })
        mlflow.set_tag("selection_rationale", SELECTION_RATIONALE)
        mlflow.set_tag("overfit_verdict",
                       metrics["overfit_report"]["verdict_train_test"])
        for artifact_path in metrics.get("artifacts", []):
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    # -----------------------------------------------------------------------
    # STEP 10 — Persist Artefacts
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 10: Saving Artefacts")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    logger.info(f"Model saved → {MODELS_DIR / 'best_model.joblib'}")

    metadata = {
        "best_model"        : MODEL_NAME,
        "best_params"       : best_params,
        "feature_names"     : all_features,
        "optimal_threshold" : metrics["optimal_threshold"],
        "calibrated"        : metrics["calibrated"],
        "cv_roc_auc"        : round(best_cv_score, 4),
        "train_roc_auc"     : metrics["train_roc_auc"],
        "val_roc_auc"       : metrics["val_roc_auc"],
        "test_roc_auc"      : metrics["test_roc_auc"],
        "test_f1_macro"     : metrics["test_f1_macro"],
        "test_mcc"          : metrics["test_mcc"],
        "test_brier_score"  : metrics["test_brier_score"],
        "delta_auc_baseline": round(delta_auc, 4),
        "overfit_verdict"   : metrics["overfit_report"]["verdict_train_test"],
        "selection_rationale": SELECTION_RATIONALE,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info(f"Metadata saved → {MODELS_DIR / 'model_metadata.json'}")

    # Final summary
    _print_final_summary(metrics, best_params, best_cv_score, baseline_auc)

    return {"best_model": MODEL_NAME, "metrics": metrics, "metadata": metadata}


# ---------------------------------------------------------------------------
# Final summary printer
# ---------------------------------------------------------------------------

def _print_final_summary(
    metrics: dict, best_params: dict, cv_auc: float, baseline_auc: float
) -> None:
    """Print a formatted final performance summary to the log."""
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("FINAL PIPELINE SUMMARY")
    logger.info(sep)
    logger.info(f"  Model            : {MODEL_NAME}")
    logger.info(f"  Baseline AUC     : {baseline_auc:.4f}")
    logger.info(f"  CV ROC-AUC       : {cv_auc:.4f}")
    logger.info(f"  Train ROC-AUC    : {metrics['train_roc_auc']:.4f}")
    logger.info(f"  Val   ROC-AUC    : {metrics['val_roc_auc']:.4f}")
    logger.info(f"  Test  ROC-AUC    : {metrics['test_roc_auc']:.4f}")
    logger.info(f"  Test  F1-macro   : {metrics['test_f1_macro']:.4f}")
    logger.info(f"  Test  MCC        : {metrics['test_mcc']:.4f}")
    logger.info(f"  Test  Brier      : {metrics['test_brier_score']:.4f}")
    logger.info(f"  Optimal threshold: {metrics['optimal_threshold']}")
    logger.info(f"  Calibrated       : {metrics['calibrated']}")
    logger.info(f"  Overfit verdict  : {metrics['overfit_report']['verdict_train_test']}")
    logger.info(sep)
    logger.info("Best hyperparameters:")
    for k, v in best_params.items():
        logger.info(f"    {k}: {v}")
    logger.info(sep)
    recs = metrics["overfit_report"].get("recommendations", [])
    if recs:
        logger.info("Recommendations:")
        for r in recs:
            logger.info(f"  💡 {r}")
    logger.info(sep)


if __name__ == "__main__":
    run_full_pipeline()
