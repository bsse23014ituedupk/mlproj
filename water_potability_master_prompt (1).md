# 🧠 MASTER PROMPT — Water Potability Classifier (End-to-End ML Project)

---

## 🎯 ROLE & OBJECTIVE

You are a **Senior ML Engineer and MLOps Architect** with deep expertise in scikit-learn, MLflow, FastAPI, and Docker. Your task is to build a **production-grade, fully documented, rubric-compliant Water Potability Classification system** from scratch using only these algorithms: **Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest**. Every decision must be **explicitly justified** in code comments and docstrings — this project will be evaluated in a live viva.

---

## 📦 DATASET FACTS (Already Analyzed — Do Not Re-Derive)

- **File:** `water_potability.csv`
- **Rows:** 3,276 | **Features:** 9 numeric float | **Target:** `Potability` (0 = Not Safe, 1 = Safe)
- **Missing values:** `ph` → 491 nulls, `Sulfate` → 781 nulls, `Trihalomethanes` → 162 nulls
- **Class imbalance:** Class 0 = 1,998 samples, Class 1 = 1,278 samples (~61/39 split)
- **Features:** `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- **All features are continuous floats** — no categorical encoding needed

---

## 📁 STEP 1 — GENERATE `requirements.txt`

Create a `requirements.txt` with **pinned versions** for full reproducibility. Include:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0          # Logistic Regression, KNN, Decision Tree, Random Forest, GridSearchCV
imbalanced-learn==0.12.3     # SMOTE for class imbalance
mlflow==2.13.0               # Experiment tracking
fastapi==0.111.0             # REST API
uvicorn[standard]==0.30.1    # ASGI server
pydantic==2.7.1              # Request/response validation
joblib==1.4.2                # Model serialization
matplotlib==3.9.0            # Plots
seaborn==0.13.2              # Plots
plotly==5.22.0               # Interactive charts
pytest==8.2.0                # Testing
httpx==0.27.0                # FastAPI test client
python-dotenv==1.0.1         # Environment variables
```

---

## 📂 STEP 2 — GENERATE FOLDER STRUCTURE

Create the following exact directory tree with a brief docstring comment in every `__init__.py` and `README.md` at root level:

```
water_potability_classifier/
│
├── data/
│   ├── raw/                        # Original unmodified CSV goes here
│   ├── processed/                  # Cleaned, imputed, encoded outputs
│   └── splits/                     # train.csv, val.csv, test.csv after splitting
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis (non-production)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All hyperparams, paths, seeds — single source of truth
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py            # Load raw CSV, validate schema
│   │   ├── preprocessing.py        # Imputation, scaling, SMOTE
│   │   └── validation.py           # Great Expectations / custom schema checks
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Interaction terms, domain-derived features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # DummyClassifier for sanity check
│   │   ├── train.py                # Main training loop with MLflow
│   │   ├── evaluate.py             # All metrics + SHAP importance
│   │   ├── optimize.py             # Optuna HPO study
│   │   └── registry.py             # MLflow Model Registry push/load
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── predict.py              # Prediction logic calling registered model
│   │   └── health.py               # /health and /ready endpoints
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Structured logging (loguru or logging)
│       └── plotting.py             # Reusable visualization helpers
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_schemas.py
│   └── integration/
│       └── test_api.py             # FastAPI TestClient end-to-end
│
├── mlruns/                         # Auto-created by MLflow — do not edit manually
├── models/                         # Serialized model artifacts (joblib)
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 STEP 3 — DATA PREPARATION (`src/data/`)

### `ingestion.py`
- Load `data/raw/water_potability.csv` using pandas
- Validate that all 9 expected feature columns are present (raise `ValueError` with clear message if not)
- Validate `Potability` is binary {0, 1}
- Log dataset shape, null counts, and class balance to MLflow as `mlflow.log_params`
- Return a typed `pd.DataFrame` with a docstring explaining each feature's **WHO/EPA significance** (pH safe range 6.5–8.5, turbidity <4 NTU, etc.)

### `preprocessing.py`
Build a **scikit-learn Pipeline** object. Steps must be in this exact order:

1. **Imputation** — Use `IterativeImputer` (MICE — Multivariate Imputation by Chained Equations) for `ph`, `Sulfate`, `Trihalomethanes`. Justify in a comment: *"KNN and median imputation ignore feature correlations; MICE models each feature as a function of others, producing less biased estimates on MAR data."*

2. **Outlier clipping** — Use `RobustScaler`-based IQR clipping (custom transformer). Justify: *"Solids and Hardness have extreme tails; clipping at 1.5×IQR before scaling prevents distortion of distance-based models."*

3. **Scaling** — Use `RobustScaler` (not `StandardScaler`). Justify: *"RobustScaler uses median/IQR and is resistant to the outliers confirmed in EDA."*

4. **Class imbalance** — Apply **SMOTE** (`imblearn.over_sampling.SMOTE`) on training split only, NEVER on validation or test. Justify: *"The 61/39 split will bias most classifiers toward majority class; SMOTE synthesizes minority-class samples in feature space without information leakage."*

All pipeline steps must be named (use `Pipeline([('imputer', ...), ('scaler', ...)])`) so they are inspectable in MLflow's artifact viewer.

### `validation.py`
Write custom assertions (or use Great Expectations) to verify:
- No nulls remain after imputation
- All feature values are within plausible physical bounds (pH ∈ [0,14], Turbidity > 0, etc.)
- Train/val/test splits are stratified (verify class ratios match)

---

## 🏗️ STEP 4 — FEATURE ENGINEERING (`src/features/`)

Create **three domain-informed derived features** with explicit chemical justification:

1. `ph_hardness_ratio = ph / (Hardness + 1e-6)` — *"Hard water often neutralizes acidity; this ratio captures joint chemical load."*
2. `tthm_chloramine_interaction = Trihalomethanes * Chloramines` — *"THMs are disinfection byproducts of chloramine reactions; their product models synergistic contamination."*
3. `organic_solids_ratio = Organic_carbon / (Solids + 1e-6)` — *"High organic carbon relative to total dissolved solids suggests biological contamination."*

Log whether these features improved validation AUC-ROC as `mlflow.log_metric("feature_eng_delta_auc", delta)`.

---

## 🤖 STEP 5 — MODEL TRAINING WITH MLFLOW (`src/models/train.py`)

### Baseline First
Always train a `DummyClassifier(strategy="most_frequent")` first and log it as the baseline run. Any final model must beat baseline by a logged margin.

### Candidate Models
Train all four in separate MLflow runs under the same **experiment name** `"water_potability_v1"`:

1. **Logistic Regression** — `LogisticRegression(class_weight="balanced", max_iter=1000)` — justify: *"Linear baseline; fast to train and highly interpretable via coefficients. class_weight='balanced' compensates for the 61/39 imbalance without oversampling."*
2. **K-Nearest Neighbours** — `KNeighborsClassifier` — justify: *"Non-parametric; makes no assumptions about feature distribution. Sensitive to scale, so RobustScaler in the pipeline is essential."*
3. **Decision Tree** — `DecisionTreeClassifier(class_weight="balanced")` — justify: *"Fully interpretable; produces human-readable if/else rules for each chemical threshold. Prone to overfitting — controlled via max_depth tuning."*
4. **Random Forest** — `RandomForestClassifier(class_weight="balanced")` — justify: *"Ensemble of Decision Trees that reduces variance via bagging. Typically the strongest performer on tabular data among class-studied models."*

### MLflow Logging (Mandatory for Each Run)
```python
with mlflow.start_run(run_name=model_name):
    mlflow.log_params({
        "model_type": ...,
        "n_features": ...,
        "smote_applied": True,
        "imputer": "IterativeImputer",
        "scaler": "RobustScaler",
        **best_hyperparams   # from GridSearchCV
    })
    mlflow.log_metrics({
        "train_roc_auc": ...,
        "val_roc_auc": ...,
        "test_roc_auc": ...,
        "test_f1_macro": ...,
        "test_precision": ...,
        "test_recall": ...,
        "test_mcc": ...,     # Matthews Correlation Coefficient
        "test_brier_score": ...
    })
    mlflow.log_artifact("reports/confusion_matrix.png")
    mlflow.log_artifact("reports/shap_summary.png")
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
```

---

## ⚙️ STEP 6 — HYPERPARAMETER OPTIMIZATION (`src/models/optimize.py`)

Use **GridSearchCV with Stratified K-Fold cross-validation** (the method studied in class).

**Cross-Validation Setup:**
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Stratified ensures each fold keeps the same 61/39 class ratio as the full dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
```

### Search Grids (one per model):

**Logistic Regression:**
```python
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],           # Inverse regularization strength
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"]
}
```

**KNN:**
```python
param_grid = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],   # Odd values avoid ties
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
```

**Decision Tree:**
```python
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}
```

**Random Forest:**
```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
}
```

**Scoring:** `scoring="roc_auc"` — justify: *"Accuracy is misleading on imbalanced data; ROC-AUC measures the model's ability to rank safe vs unsafe samples regardless of threshold."*

**K-Means Bonus Analysis:** Run `KMeans(n_clusters=2)` on the feature matrix (unsupervised) and compare its cluster assignments to the true `Potability` labels. Log the Adjusted Rand Index to MLflow as `mlflow.log_metric("kmeans_ari", ari)`. Justify: *"K-Means reveals whether chemical profiles naturally separate into two groups matching potability — if ARI is high, the classes are linearly separable in feature space."*

**Model Selection Criterion (explicit — critical for viva):**
> *"All four models are compared by their mean 5-fold CV ROC-AUC on the training set. The model with the highest test-set ROC-AUC wins. If two models are within 0.005 AUC, the one with higher F1-Macro wins (penalises poor recall on the minority safe-water class). This is important because a false negative — predicting unsafe water is safe — is far more dangerous than a false positive."*

Log this rationale as `mlflow.set_tag("selection_rationale", "...")`.

---

## 📊 STEP 7 — EVALUATION (`src/models/evaluate.py`)

Produce all of the following on the **held-out test set** (20% stratified split, set aside before any preprocessing fit):

### Metrics
- ROC-AUC, PR-AUC (preferred for imbalanced data — justify in comment)
- F1-Macro, F1-Weighted, Precision, Recall per class
- Matthews Correlation Coefficient (MCC) — justify: *"MCC is the only metric robust to class imbalance that considers all four confusion matrix quadrants"*
- Brier Score (probability calibration quality)
- Cohen's Kappa

### Visualizations (all saved to `reports/` and logged to MLflow)
1. **Confusion Matrix** — with absolute counts and row-normalized percentages
2. **ROC Curve** — with AUC annotation, compared against baseline
3. **Precision-Recall Curve** — with Average Precision annotation
4. **SHAP Summary Plot** (beeswarm) — *"SHAP values quantify each feature's marginal contribution to the prediction using Shapley values from cooperative game theory"*
5. **SHAP Bar Plot** — global mean |SHAP| feature importance
6. **Calibration Curve** — to justify confidence in probability outputs sent via API
7. **Learning Curve** — to detect overfitting/underfitting

### Dangerous Chemical Factors Report
Using SHAP mean absolute values, programmatically generate a ranked Markdown table:
```
| Rank | Feature          | Mean |SHAP| | Health Significance             |
|------|------------------|-------------|---------------------------------|
| 1    | Sulfate          | 0.XXX       | Laxative effect >500mg/L        |
| 2    | ph               | 0.XXX       | WHO safe range 6.5–8.5          |
...
```
Save this as `reports/dangerous_factors.md` and log as MLflow artifact.

---

## 🌐 STEP 8 — FASTAPI APPLICATION (`src/api/`)

### `schemas.py`
```python
class WaterSample(BaseModel):
    ph: float = Field(..., ge=0.0, le=14.0, description="pH level (WHO: 6.5-8.5)")
    Hardness: float = Field(..., gt=0, description="Calcium carbonate mg/L")
    Solids: float = Field(..., gt=0, description="Total dissolved solids ppm")
    Chloramines: float = Field(..., ge=0, description="Chloramines ppm")
    Sulfate: float = Field(..., ge=0, description="Sulfate mg/L")
    Conductivity: float = Field(..., gt=0, description="Electrical conductivity μS/cm")
    Organic_carbon: float = Field(..., ge=0, description="Organic carbon ppm")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes μg/L")
    Turbidity: float = Field(..., gt=0, description="Turbidity NTU")

class PredictionResponse(BaseModel):
    potability: int                   # 0 or 1
    label: str                        # "Safe" or "Unsafe"
    confidence: float                 # Probability of predicted class
    risk_level: str                   # "Low" / "Medium" / "High"
    top_risk_factors: list[str]       # Top 3 SHAP contributors for this sample
    model_version: str                # MLflow run ID
    prediction_id: str                # UUID for tracing
```

### `main.py` — Endpoints
- `GET /health` → returns `{"status": "ok", "model_loaded": true}`
- `GET /ready` → returns model metadata (version, training date, test AUC)
- `POST /predict` → single sample prediction with full `PredictionResponse`
- `POST /predict/batch` → list of samples, returns list of `PredictionResponse`
- `GET /model/info` → returns feature importances and model selection rationale
- `GET /docs` → auto-generated Swagger UI (built-in FastAPI)

### Model Loading Strategy
Load the **best registered MLflow model** at startup using:
```python
model = mlflow.sklearn.load_model(f"models:/water_potability_classifier/Production")
```
Cache it in a module-level variable. Add lifespan startup/shutdown handlers.

### Per-Prediction SHAP
For each `/predict` call, compute **local SHAP values** for that single sample and return the top 3 features driving the prediction (with direction: "↑ increases risk" or "↓ decreases risk").

---

## 🐳 STEP 9 — CONTAINERIZATION (`Dockerfile` + `docker-compose.yml`)

### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`
Define three services:
1. **`api`** — the FastAPI app on port 8000
2. **`mlflow`** — MLflow tracking server on port 5000, with a volume-mounted `mlruns/`
3. **`trainer`** — one-shot training container that runs the full pipeline and exits

Use environment variables from `.env.example` for all secrets and paths.

---

## 🧪 STEP 10 — TESTING (`tests/`)

### Unit Tests
- `test_preprocessing.py`: Assert imputer removes all nulls; assert SMOTE is NOT applied to test set; assert scaler output has no infinite values
- `test_features.py`: Assert derived features are finite; assert no data leakage (fit only on train)
- `test_schemas.py`: Assert invalid pH (>14) raises `422 Unprocessable Entity`

### Integration Tests (`test_api.py`)
Using FastAPI `TestClient`:
- `test_predict_valid_sample()` — POST a valid sample, assert response has all required fields
- `test_predict_invalid_ph()` — POST pH=15, assert 422 response
- `test_health_endpoint()` — GET /health returns 200
- `test_batch_predict()` — POST 10 samples, assert 10 responses returned

---

## 📝 STEP 11 — CODE QUALITY REQUIREMENTS

Every single file must meet these standards (they will be checked in viva):

1. **Every function has a Google-style docstring** with Args, Returns, Raises sections
2. **Every non-obvious line has an inline comment** explaining *why*, not *what*
3. **All magic numbers are named constants** in `config.py` (e.g., `RANDOM_SEED = 42`, `TEST_SIZE = 0.20`, `SMOTE_K_NEIGHBORS = 5`)
4. **No raw string paths** — use `pathlib.Path` everywhere
5. **No data leakage** — pipeline `fit()` called ONLY on training data
6. **Type hints** on all function signatures
7. **Model selection is explicitly logged and explainable** — MLflow `set_tag("selection_rationale", ...)`

---

## 🏁 EXECUTION ORDER

Generate files in this exact order:
1. `requirements.txt`
2. Folder structure (all `__init__.py` and empty files with docstrings)
3. `src/config.py`
4. `src/utils/logger.py`
5. `src/data/ingestion.py`
6. `src/data/preprocessing.py`
7. `src/data/validation.py`
8. `src/features/feature_engineering.py`
9. `src/models/baseline.py`
10. `src/models/optimize.py`
11. `src/models/train.py`
12. `src/models/evaluate.py`
13. `src/models/registry.py`
14. `src/api/schemas.py`
15. `src/api/predict.py`
16. `src/api/health.py`
17. `src/api/main.py`
18. `tests/unit/test_preprocessing.py`
19. `tests/unit/test_schemas.py`
20. `tests/integration/test_api.py`
21. `Dockerfile`
22. `docker-compose.yml`
23. `.env.example`
24. `README.md`

---

## ⚠️ HARD CONSTRAINTS

- **Never fit any transformer on validation or test data** — this is grounds for immediate failure in viva
- **Never use `accuracy` as the primary metric** — dataset is imbalanced; use ROC-AUC and F1-Macro
- **Every model choice must have a written justification** — "I used XGBoost because it supports native handling of class imbalance via `scale_pos_weight`, is robust to feature scale, and has a proven track record on tabular chemical sensor data"
- **MLflow must track every experiment** — no bare `model.fit()` calls outside a `with mlflow.start_run()` block
- **SHAP explanations are mandatory** — the rubric explicitly requires identifying the most dangerous chemical factors

---

## 🎤 VIVA PREPARATION (Pre-generate Answers)

At the end of code generation, produce a `VIVA_QA.md` file with confident, technically precise answers to:

1. Why did you choose Random Forest as your best model over Logistic Regression and KNN?
2. How did you handle missing values and why MICE over median imputation?
3. What is SMOTE and why did you apply it only on training data?
4. What is K-Fold Cross Validation and why did you use Stratified K-Fold specifically?
5. Why is ROC-AUC better than accuracy for this dataset?
6. How does a Decision Tree decide which feature to split on first?
7. What is the difference between a Decision Tree and a Random Forest?
8. What does K-Means clustering tell you about this dataset, and what is Adjusted Rand Index?
9. Why did you use `class_weight="balanced"` and what does it do?
10. If pH is the most important feature in Random Forest, what does that mean physically for water safety?
