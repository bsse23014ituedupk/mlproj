"""
Hyperparameter Optimization Module — GridSearchCV with K-Fold.

Uses GridSearchCV with KFold to run SMOTE inside each CV fold (via imblearn
Pipeline), preventing data leakage. Scoring is f1_macro — balanced metric
for imbalanced data.

Also includes K-Means bonus analysis comparing unsupervised clusters
against true potability labels (Adjusted Rand Index).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_SEED, CV_N_SPLITS, CV_SCORING,
    RANDOM_FOREST_GRID, SMOTE_K_NEIGHBORS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)



def get_model_and_grid(model_name: str) -> tuple:
    """Return the model instance and its hyperparameter search grid.

    Args:
        model_name: Must be 'RandomForest'.

    Returns:
        Tuple of (pipeline_instance, param_grid_dict).
    """
    if model_name != "RandomForest":
        raise ValueError("Only 'RandomForest' is supported.")

    # Use imblearn Pipeline to apply SMOTE *within* each CV fold
    # This prevents data leakage and over-optimistic CV scores.
    pipeline = Pipeline([
        ("smote", SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    # Prefix grid parameters with 'model__' to target the estimator in the pipeline
    param_grid = {f"model__{k}": v for k, v in RANDOM_FOREST_GRID.items()}

    return pipeline, param_grid


def run_grid_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """Run GridSearchCV for a given model.

    Scoring: ROC-AUC — "Accuracy is misleading on imbalanced data; ROC-AUC
    measures the model's ability to rank safe vs unsafe samples regardless
    of threshold."

    Args:
        model_name: Name of the model to optimize.
        X_train: Training features (after SMOTE).
        y_train: Training labels (after SMOTE).

    Returns:
        Fitted GridSearchCV object with best parameters.
    """
    model, param_grid = get_model_and_grid(model_name)

    logger.info(f"Starting GridSearchCV for {model_name} "
                f"({CV_N_SPLITS}-fold, scoring={CV_SCORING})")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=CV_SCORING,
        cv=KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        verbose=1,
        refit=True,  # Refit best model on full training data
    )

    grid_search.fit(X_train, y_train)

    logger.info(
        f"{model_name} — Best CV {CV_SCORING}: {grid_search.best_score_:.4f}"
    )
    logger.info(f"{model_name} — Best params: {grid_search.best_params_}")

    return grid_search


def run_kmeans_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Run K-Means (k=2) and compute Adjusted Rand Index vs true labels.

    Justification: K-Means reveals whether chemical profiles naturally
    separate into two groups matching potability. If ARI is high, the
    classes are linearly separable in feature space and simpler models
    may suffice. Low ARI confirms that non-linear models (Random Forest)
    are needed.

    Args:
        X_train: Training feature matrix (preprocessed).
        y_train: True potability labels.

    Returns:
        Adjusted Rand Index (float between -1 and 1; 1 = perfect).
    """
    logger.info("Running K-Means bonus analysis (k=2)...")

    # Cluster without using labels — purely unsupervised
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)

    # Compare cluster assignments to true potability labels
    # ARI adjusts for chance — 0 = random, 1 = perfect agreement
    ari = adjusted_rand_score(y_train, cluster_labels)

    logger.info(
        f"K-Means ARI vs Potability labels: {ari:.4f} "
        f"({'High — classes are separable' if ari > 0.1 else 'Low — non-linear boundary needed'})"
    )
    return ari
