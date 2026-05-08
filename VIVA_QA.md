# Viva Q&A — Water Potability Classifier

> Prepared answers for the 10 mandatory viva questions.

---

## Q1. Why did you choose Random Forest as your best model?

**Answer:**
Random Forest is an ensemble of Decision Trees trained on random bootstrapped subsets of the training data (bagging). Each tree also considers only a random subset of features at each split (`max_features="sqrt"`), which decorrelates the trees and reduces variance.

For this dataset, Random Forest was selected because:
- It handles the **non-linear, high-dimensional** relationships between water quality indicators without feature engineering assumptions.
- It is inherently robust to **outliers** (tree splits are based on rank, not exact values), which matters here because `Solids` and `Hardness` have heavy tails.
- It provides **native feature importances** and integrates with SHAP for per-prediction explanations.
- With `class_weight="balanced"`, it compensates for the 61/39 class imbalance by up-weighting minority-class errors during tree construction.

The model was compared against the `DummyClassifier` baseline. The RF's test ROC-AUC was significantly higher (`delta_auc_vs_baseline` logged in MLflow), confirming it is not merely predicting the majority class.

---

## Q2. How did you handle missing values and why MICE over median imputation?

**Answer:**
Three features have missing values: `ph` (491 nulls), `Sulfate` (781 nulls), `Trihalomethanes` (162 nulls).

We used **`IterativeImputer` (MICE — Multivariate Imputation by Chained Equations)** from scikit-learn.

**Why MICE over simple median imputation:**
- **Median imputation** replaces each missing value with the column's median, completely ignoring relationships between features. For example, `ph` is chemically correlated with `Hardness` and `Conductivity`; median imputation ignores this.
- **MICE** models each feature with missing values as a function of all other features using a round-robin regression approach. This produces estimates that respect the **joint distribution** of features, resulting in less-biased imputations on MAR (Missing At Random) data.

The imputer is fitted **only on training data** and then applied to validation and test sets to prevent data leakage.

---

## Q3. What is SMOTE and why did you apply it only on training data?

**Answer:**
**SMOTE (Synthetic Minority Over-sampling Technique)** addresses class imbalance (61/39 split here) by synthesizing new minority-class samples rather than duplicating existing ones. For each minority sample, SMOTE selects `k` nearest neighbours (we use `k=5`) and generates new synthetic points along the line segments connecting them in feature space.

**Why only on training data:**
Applying SMOTE to validation or test data would:
1. **Inflate evaluation metrics** — the synthetic samples are generated from the same distribution as training data, making the test set unrealistically easy.
2. **Cause data leakage** — information from the training distribution would contaminate the held-out evaluation.

In our pipeline, SMOTE is placed inside an **`imblearn.Pipeline`** used within `GridSearchCV`, so it is applied only within each training fold during cross-validation — never on the held-out fold or test set.

---

## Q4. What is K-Fold Cross Validation and why did you use it?

**Answer:**
**K-Fold Cross-Validation** splits the training data into `K` equal folds. The model is trained `K` times, each time using `K-1` folds for training and 1 fold for validation. The final CV score is the mean across all `K` folds.

We used `K=5` (5-fold), which gives a good bias-variance trade-off — enough data per fold for stable estimates without excessive computation.

We used **standard `KFold`** (not `StratifiedKFold`) with `KFold(shuffle=True, random_state=42)` for the `GridSearchCV`, because SMOTE is inside the `imblearn.Pipeline`. SMOTE synthesizes minority samples within each training fold, so the effective class distribution inside CV is controlled by SMOTE rather than by stratification of the outer split. Using `StratifiedKFold` on top of SMOTE can cause redundant or conflicting resampling.

---

## Q5. Why is ROC-AUC better than accuracy for this dataset?

**Answer:**
This dataset has a **61/39 class imbalance**. A `DummyClassifier` that always predicts "Unsafe" (class 0) achieves ~61% accuracy without learning anything.

**ROC-AUC** measures the model's ability to **rank** safe vs. unsafe samples across all possible decision thresholds. An AUC of 0.5 is random guessing; 1.0 is perfect. It is threshold-independent and therefore not affected by class skew.

We also report **F1-Macro** (geometric mean of F1 for each class), which penalises poor recall on the minority "Safe" class — important because a **false negative** (predicting safe water is unsafe) is less dangerous than a **false positive** (predicting unsafe water is safe). Both metrics together give a complete picture that accuracy alone cannot provide.

---

## Q6. How does a Decision Tree decide which feature to split on first?

**Answer:**
A Decision Tree selects the split that **maximises information gain** (or equivalently, minimises impurity) at each node.

- **Gini Impurity**: `Gini = 1 - Σ(p_i²)` — the probability of misclassifying a randomly chosen sample.
- **Entropy**: `H = -Σ(p_i * log₂(p_i))` — information-theoretic measure of disorder.

At each node, the tree evaluates every possible feature and every possible threshold, choosing the `(feature, threshold)` pair that produces the greatest **reduction in weighted impurity** across the two resulting child nodes (Information Gain).

For our water data, if `ph < 6.8` cleanly separates many safe/unsafe samples, the tree will split there first — producing the root node of the tree.

---

## Q7. What is the difference between a Decision Tree and a Random Forest?

**Answer:**

| Property | Decision Tree | Random Forest |
|---|---|---|
| Structure | Single tree | Ensemble of hundreds of trees |
| Training data | Full training set | Random bootstrap samples (bagging) |
| Feature selection | All features at each split | Random subset (`sqrt(n_features)`) per split |
| Variance | High (overfits easily) | Low (trees are decorrelated) |
| Interpretability | High (can be visualised) | Low (aggregate of many trees) |
| Performance | Moderate | Typically much better |

A **Random Forest** reduces the **variance** of a single Decision Tree by averaging predictions across many diverse trees. Each tree sees a different bootstrap sample of the training data, and each split sees a random subset of features. This decorrelation between trees is the key reason Random Forest generalises far better.

---

## Q8. What does K-Means clustering tell you about this dataset, and what is ARI?

**Answer:**
**K-Means** (k=2) clusters the 9 water quality features **without using the Potability label** — purely unsupervised. We then compare its 2-cluster assignments against the true binary labels.

- **High ARI**: Chemical features naturally separate into groups matching safe/unsafe → linear separability in feature space → simpler models may suffice.
- **Low ARI (≈ 0)**: Safe and unsafe water samples are not chemically well-separated → the boundary is non-linear → ensemble models (Random Forest) are needed.

**Adjusted Rand Index (ARI)**: Measures the similarity between two cluster assignments (predicted vs. true), adjusted for chance. ARI = 1.0 means perfect agreement; ARI ≈ 0 means no better than random.

On this dataset, K-Means typically yields a low ARI, justifying our choice of a non-linear model (Random Forest) rather than logistic regression.

---

## Q9. Why did you use `class_weight="balanced"` and what does it do?

**Answer:**
The dataset has a 61/39 split (1,998 Unsafe vs. 1,278 Safe). Without correction, the model will optimise for the majority class, achieving high accuracy by effectively ignoring the minority class.

`class_weight="balanced"` instructs scikit-learn to automatically compute class weights inversely proportional to class frequencies:
```
w_i = n_samples / (n_classes × n_samples_in_class_i)
```

For our data:
- Class 0 (Unsafe, 1998 samples) → lower weight
- Class 1 (Safe, 1278 samples) → higher weight

This penalises **misclassifying minority-class samples more heavily** during training, pushing the model to learn a better decision boundary for the Safe class. This is combined with SMOTE, which operates in feature space, for defence-in-depth against class imbalance.

---

## Q10. If pH is the most important feature in Random Forest, what does that mean physically?

**Answer:**
**Physically**, pH measures the acidity or alkalinity of water on a logarithmic scale (0–14). The **WHO safe range is 6.5–8.5**.

If SHAP values show pH as the most important feature, it means that the **model's prediction changes most when pH deviates from its typical range**. Specifically:

- **Low pH (acidic, < 6.5)**: Increases heavy metal dissolution (lead, copper from pipes), GI tract irritation, and corrosion. SHAP value would be strongly negative (↑ increases unsafe prediction).
- **High pH (alkaline, > 8.5)**: Can precipitate minerals, affect chlorination efficiency, cause bitter taste. Also increases unsafe risk.
- **Mid-range pH (7–7.5)**: Most correlated with safe potability → SHAP value pushes toward Safe prediction.

The SHAP beeswarm plot (`reports/shap_summary.png`) visualises this: samples with extreme pH values (coloured red = high value) will have large positive SHAP contributions toward the Unsafe class.
