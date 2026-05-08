# 💧 Water Potability Classifier

A **production-grade, end-to-end Machine Learning system** for predicting water potability using chemical quality indicators. Built with scikit-learn, MLflow, FastAPI, and Docker.

## 🎯 Problem Statement

Given 9 chemical water quality parameters, classify whether a water sample is **Safe (1)** or **Unsafe (0)** for human consumption based on WHO/EPA guidelines.

## 📊 Dataset

- **Source:** [Kaggle - Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Rows:** 3,276 | **Features:** 9 numeric float | **Target:** `Potability` (binary)
- **Class Distribution:** 61% Unsafe / 39% Safe (imbalanced)
- **Missing Values:** `ph` (491), `Sulfate` (781), `Trihalomethanes` (162)

## 🏗️ Architecture

```
src/
├── config.py              # Central configuration (all constants)
├── data/                  # Ingestion, preprocessing, validation
├── features/              # Domain-informed feature engineering
├── models/                # Training, evaluation, optimization, registry
├── api/                   # FastAPI serving layer
└── utils/                 # Logging, plotting helpers
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python -c "from src.data.ingestion import download_dataset; download_dataset()"
```

### 3. Train Models
```bash
python -m src.models.train
```

### 4. Start API Server
```bash
# Set PYTHONPATH to current directory and run uvicorn as a module
$env:PYTHONPATH="."; python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the **Interactive API Docs (Swagger UI)** at:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ph": 7.0, "Hardness": 200, "Solids": 20000, "Chloramines": 7, "Sulfate": 330, "Conductivity": 400, "Organic_carbon": 14, "Trihalomethanes": 60, "Turbidity": 3.5}'
```

## 🐳 Docker

```bash
docker-compose up --build
```

Services:
- **API**: http://localhost:8000 (Swagger UI at /docs)
- **MLflow**: [http://localhost:5000](http://localhost:5000)
  ```bash
  # Start UI manually
  python -m mlflow ui --port 5000
  ```

## 🤖 Models

| Model | Description | Justification |
|-------|-------------|---------------|
| Logistic Regression | Linear baseline | Fast, interpretable via coefficients |
| KNN | Non-parametric | No distribution assumptions |
| Decision Tree | Rule-based | Human-readable if/else rules |
| Random Forest | Ensemble | Reduces variance via bagging |

## 📈 Evaluation Metrics

- **Primary:** ROC-AUC (robust to class imbalance)
- **Secondary:** F1-Macro, MCC, PR-AUC, Brier Score, Cohen's Kappa
- **Explainability:** SHAP values for feature importance

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Key Design Decisions

1. **MICE Imputation** over median — models feature correlations
2. **RobustScaler** over StandardScaler — resistant to outliers
3. **SMOTE** on training data only — prevents information leakage
4. **Stratified K-Fold** — preserves class ratio in each fold
5. **ROC-AUC** over accuracy — meaningful for imbalanced data

## 👥 Author

Water Potability Classifier Project — ML Engineering
