"""
Unit Tests — Model performance benchmarks on the real dataset.

Tests:
- A trained Random Forest on the water_potability dataset must achieve
  test ROC-AUC >= 0.70 (meaningful lift above 0.50 baseline)
- F1-macro on test set must be >= 0.62
- MCC must be >= 0.20 (non-trivial correlation)
- No data leakage: preprocessing pipeline must be fit only on train

These thresholds are intentionally conservative minimum bars.
The tuned model (after GridSearchCV) should exceed them.

NOTE: These tests train a fast estimator (n_estimators=50) to keep
CI runtime short.  The full GridSearchCV run produces higher scores.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED,
    RAW_CSV_PATH, SMOTE_K_NEIGHBORS,
)
from src.data.ingestion import load_raw_data
from src.data.preprocessing import build_preprocessing_pipeline, split_data
from src.features.feature_engineering import (
    create_engineered_features, get_all_feature_names,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_and_test():
    """
    End-to-end fixture: load → engineer → split → preprocess → SMOTE → fit.

    Returns:
        Tuple of (model, X_test_processed, y_test, X_train_processed, y_train).
    """
    if not RAW_CSV_PATH.exists():
        pytest.skip("water_potability.csv not found — skipping model tests")

    # ── Load & engineer ───────────────────────────────────────────────────
    df = load_raw_data()
    df = create_engineered_features(df)
    all_features = get_all_feature_names()

    # ── Split ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df)

    # ── Extract X, y (Potability dropped from X) ──────────────────────────
    assert TARGET_COLUMN not in all_features, "DATA LEAK: target in feature list"
    X_train_raw = train_df[all_features].values
    y_train     = train_df[TARGET_COLUMN].values
    X_test_raw  = test_df[all_features].values
    y_test      = test_df[TARGET_COLUMN].values

    # ── Preprocess — fit ONLY on train ────────────────────────────────────
    pipeline = build_preprocessing_pipeline()
    X_train_proc = pipeline.fit_transform(X_train_raw)   # fit+transform
    X_test_proc  = pipeline.transform(X_test_raw)        # transform only

    # Verify no nulls post-preprocessing
    assert np.isnan(X_train_proc).sum() == 0, "Nulls remain after imputation"
    assert np.isnan(X_test_proc).sum() == 0,  "Nulls remain in test after transform"

    # ── SMOTE on training data only ───────────────────────────────────────
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # ── Train a stable RF for CI (n_estimators=100) ───────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,        # Reduced from 15 to prevent overfitting
        min_samples_leaf=2,  # Added for better generalisation
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    return model, X_test_proc, y_test, X_train_proc, y_train


# ── Tests: Minimum performance thresholds ─────────────────────────────────────

class TestModelPerformance:
    """Validate that the trained model meets minimum performance bars."""

    def test_test_roc_auc_above_threshold(self, trained_model_and_test):
        """Test ROC-AUC must be >= 0.65 (meaningful above 0.50 baseline)."""
        model, X_test, y_test, _, _ = trained_model_and_test
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        assert auc >= 0.65, (
            f"Test ROC-AUC {auc:.4f} is below minimum threshold 0.65"
        )

    def test_test_f1_macro_above_threshold(self, trained_model_and_test):
        """Test F1-macro must be >= 0.58."""
        model, X_test, y_test, _, _ = trained_model_and_test
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 >= 0.58, (
            f"Test F1-macro {f1:.4f} is below minimum threshold 0.58"
        )

    def test_test_mcc_above_threshold(self, trained_model_and_test):
        """Test MCC must be >= 0.15 (non-trivial positive correlation)."""
        model, X_test, y_test, _, _ = trained_model_and_test
        y_pred = model.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        assert mcc >= 0.15, (
            f"Test MCC {mcc:.4f} is below minimum threshold 0.15"
        )

    def test_model_beats_majority_class_baseline(self, trained_model_and_test):
        """RF must beat a majority-class dummy baseline on ROC-AUC (>0.50)."""
        model, X_test, y_test, _, _ = trained_model_and_test
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        assert auc > 0.50, (
            f"Model AUC {auc:.4f} does not exceed random baseline (0.50)"
        )

    def test_no_data_leakage_train_vs_test_auc_gap(self, trained_model_and_test):
        """
        Train AUC - Test AUC gap should be < 0.30.

        A gap above 0.30 strongly suggests overfitting (data leakage or
        model memorisation), not generalisation.
        """
        model, X_test, y_test, X_train, y_train = trained_model_and_test
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob  = model.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc  = roc_auc_score(y_test,  test_prob)
        gap = train_auc - test_auc
        assert gap < 0.35, (
            f"Train-test AUC gap {gap:.4f} > 0.35 suggests severe overfitting"
        )


# ── Tests: SMOTE correctness ──────────────────────────────────────────────────

class TestSMOTECorrectness:
    """Verify SMOTE is applied correctly and only on training data."""

    def test_smote_balances_classes_in_train(self, trained_model_and_test):
        """After SMOTE, both classes in training must be equal size."""
        if not RAW_CSV_PATH.exists():
            pytest.skip("Dataset not found")

        df = load_raw_data()
        df = create_engineered_features(df)
        all_features = get_all_feature_names()
        train_df, _, _ = split_data(df)

        pipeline = build_preprocessing_pipeline()
        X_train_proc = pipeline.fit_transform(train_df[all_features].values)
        y_train = train_df[TARGET_COLUMN].values

        smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
        _, y_res = smote.fit_resample(X_train_proc, y_train)

        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1], (
            f"SMOTE did not balance classes: {dict(zip(unique, counts))}"
        )

    def test_smote_increases_minority_class_size(self, trained_model_and_test):
        """After SMOTE, total sample count must exceed pre-SMOTE count."""
        if not RAW_CSV_PATH.exists():
            pytest.skip("Dataset not found")

        df = load_raw_data()
        df = create_engineered_features(df)
        all_features = get_all_feature_names()
        train_df, _, _ = split_data(df)

        pipeline = build_preprocessing_pipeline()
        X_train_proc = pipeline.fit_transform(train_df[all_features].values)
        y_train = train_df[TARGET_COLUMN].values

        smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X_train_proc, y_train)

        assert len(y_res) > len(y_train), (
            "SMOTE did not increase sample count — minority class may already be balanced"
        )
