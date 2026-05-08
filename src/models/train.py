# Trains Random Forest model in an MLflow run under experiment 'water_potability_v1'.
# """
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED,
    MLFLOW_EXPERIMENT_NAME, MODELS_DIR,
)
from src.data.ingestion import load_raw_data, log_dataset_metadata, download_dataset
from src.data.preprocessing import (
    build_preprocessing_pipeline, split_data,
)
from src.data.validation import (
    assert_no_nulls, assert_stratified_splits,
    validate_processed_array,
)
from src.features.feature_engineering import (
    create_engineered_features, get_all_feature_names,
)
from src.models.baseline import run_baseline
from src.models.optimize import run_grid_search
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

SELECTION_RATIONALE = (
    "Random Forest is used as the primary model. It is an ensemble of "
    "Decision Trees that reduces variance via bagging, typically performing "
    "well on tabular data."
)

MODEL_NAMES = ["RandomForest"]


def run_full_pipeline() -> dict:
    """Execute the complete training pipeline end-to-end.

    Returns:
        Dict with best model name, metrics, and all run results.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Step 1: Data Ingestion -----------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    download_dataset()
    df = load_raw_data()

    # --- Step 2: Feature Engineering (before split, on raw DataFrame) ---------
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    df = create_engineered_features(df)
    all_features = get_all_feature_names()

    # --- Step 3: Train/Val/Test Split -----------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Splitting Data")
    train_df, val_df, test_df = split_data(df)
    assert_stratified_splits(train_df, val_df, test_df)

    X_train_raw = train_df[all_features].values
    y_train = train_df[TARGET_COLUMN].values
    X_val_raw = val_df[all_features].values
    y_val = val_df[TARGET_COLUMN].values
    X_test_raw = test_df[all_features].values
    y_test = test_df[TARGET_COLUMN].values

    # --- Step 4: Preprocessing (fit on train ONLY) ----------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocessing")
    pipeline = build_preprocessing_pipeline()

    # fit_transform on training data only — prevents data leakage
    X_train_processed = pipeline.fit_transform(X_train_raw)
    # transform (NOT fit) on val and test — critical for no data leakage
    X_val_processed = pipeline.transform(X_val_raw)
    X_test_processed = pipeline.transform(X_test_raw)

    validate_processed_array(X_train_processed, "train")
    validate_processed_array(X_val_processed, "val")
    validate_processed_array(X_test_processed, "test")

    # Save the preprocessing pipeline for API use
    joblib.dump(pipeline, MODELS_DIR / "preprocessing_pipeline.joblib")


    # --- Step 5a: Baseline (DummyClassifier) — must be beaten by RF -----------
    logger.info("=" * 60)
    logger.info("STEP 5a: Baseline DummyClassifier")
    baseline_metrics = run_baseline(
        X_train_processed, y_train,
        X_test_processed, y_test,
    )
    baseline_auc = baseline_metrics["test_roc_auc"]
    logger.info(f"Baseline AUC floor: {baseline_auc:.4f} — RF must exceed this")

    # --- Step 5b: Train Random Forest Model -----------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5b: Training Random Forest Model")

    model_name = "RandomForest"
    logger.info(f"Training: {model_name}")

    # GridSearchCV with cross-validation (SMOTE inside)
    grid_search = run_grid_search(model_name, X_train_processed, y_train)
    best_pipeline = grid_search.best_estimator_
    best_model = best_pipeline.named_steps["model"]
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Evaluate on val and test sets
    metrics = evaluate_model(
        best_model, X_train_processed, y_train,
        X_val_processed, y_val,
        X_test_processed, y_test,
        model_name, all_features,
    )

    # Log everything to MLflow
    with mlflow.start_run(run_name=model_name):
        log_dataset_metadata(df)
        mlflow.log_params({
            "model_type": model_name,
            "n_features": len(all_features),
            "smote_applied": True,
            "imputer": "IterativeImputer",
            "scaler": "RobustScaler",
            **best_params,
        })
        mlflow.log_metrics({
            # CV scoring is f1_macro (config.CV_SCORING) — not roc_auc
            "cv_f1_macro": best_cv_score,
            "train_roc_auc": metrics["train_roc_auc"],
            "val_roc_auc": metrics["val_roc_auc"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_f1_macro": metrics["test_f1_macro"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_mcc": metrics["test_mcc"],
            "test_brier_score": metrics["test_brier_score"],
            # Margin by which RF beats the DummyClassifier baseline
            "delta_auc_vs_baseline": metrics["test_roc_auc"] - baseline_auc,
        })
        if metrics.get("feature_eng_delta_auc") is not None:
            mlflow.log_metric("feature_eng_delta_auc",
                                metrics["feature_eng_delta_auc"])
        mlflow.set_tag("selection_rationale", SELECTION_RATIONALE)
        mlflow.set_tag("model_category", "candidate")

        # Log artifacts (plots)
        for artifact_path in metrics.get("artifacts", []):
            mlflow.log_artifact(str(artifact_path))

        # Log the model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    all_results = {
        model_name: {
            "cv_f1_macro": best_cv_score,
            "test_roc_auc": metrics["test_roc_auc"],
            "test_f1_macro": metrics["test_f1_macro"],
            "best_params": best_params,
            "model": best_model,
            "metrics": metrics,
        }
    }

    # --- Step 7: Save Model ---------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Saving Model")
    best_name = "RandomForest"
    best_model_obj = all_results[best_name]["model"]

    # Save the best model
    joblib.dump(best_model_obj, MODELS_DIR / "best_model.joblib")
    logger.info(f"Model: {best_name} saved to {MODELS_DIR / 'best_model.joblib'}")

    # Save metadata
    import json
    metadata = {
        "best_model": best_name,
        "test_roc_auc": all_results[best_name]["test_roc_auc"],
        "test_f1_macro": all_results[best_name]["test_f1_macro"],
        "best_params": all_results[best_name]["best_params"],
        "feature_names": all_features,
        "selection_rationale": SELECTION_RATIONALE,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {"best_model": best_name, "results": all_results}




if __name__ == "__main__":
    run_full_pipeline()
