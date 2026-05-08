"""
Baseline Model Module — DummyClassifier sanity check.

Trains a DummyClassifier(strategy='most_frequent') as the baseline run.
Any trained model must beat this baseline by a logged margin.

Justification: A model that always predicts the majority class achieves
~61% accuracy on this dataset. Any ML model that cannot beat this is
useless. The baseline sets the floor for acceptable performance.
"""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef,
)

from src.config import MLFLOW_EXPERIMENT_NAME, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASELINE_RATIONALE = (
    "DummyClassifier(strategy='most_frequent') always predicts the majority "
    "class (0 = Unsafe, ~61% of samples). This sets the performance floor — "
    "any trained model must exceed this to demonstrate learning."
)


def run_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train and evaluate a DummyClassifier baseline.

    Logs all metrics and tags to MLflow under the experiment
    'water_potability_v1' with run name 'Baseline_DummyClassifier'.

    Args:
        X_train: Training feature matrix (preprocessed).
        y_train: Training labels.
        X_test: Test feature matrix (preprocessed).
        y_test: Test labels.

    Returns:
        Dict with baseline metrics (test_roc_auc, test_f1_macro, etc.).
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("=" * 60)
    logger.info("BASELINE: Training DummyClassifier (most_frequent strategy)")

    # most_frequent: always predicts class 0 (majority = 'Unsafe')
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)

    # ROC-AUC requires probability estimates; DummyClassifier provides them
    y_prob = dummy.predict_proba(X_test)[:, 1]

    metrics = {
        "test_roc_auc":   roc_auc_score(y_test, y_prob),
        "test_f1_macro":  f1_score(y_test, y_pred, average="macro", zero_division=0),
        "test_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "test_recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "test_mcc":       matthews_corrcoef(y_test, y_pred),
    }

    logger.info(
        f"Baseline — AUC={metrics['test_roc_auc']:.4f}, "
        f"F1={metrics['test_f1_macro']:.4f}, MCC={metrics['test_mcc']:.4f}"
    )

    # Log to MLflow so the baseline run appears alongside trained models
    with mlflow.start_run(run_name="Baseline_DummyClassifier"):
        mlflow.log_params({
            "model_type":    "DummyClassifier",
            "strategy":      "most_frequent",
            "smote_applied": False,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_category", "baseline")
        mlflow.set_tag("selection_rationale", BASELINE_RATIONALE)
        mlflow.sklearn.log_model(dummy, artifact_path="model")

    logger.info("Baseline run logged to MLflow")
    return metrics
