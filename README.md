#  Diabetes Prediction — Optimized Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.96%25-brightgreen)]()
[![AUC](https://img.shields.io/badge/AUC-0.9539-brightgreen)]()

> An end-to-end machine learning pipeline for early diabetes detection using the PIMA Indians Diabetes Dataset. Achieves **88.96% test accuracy** and **AUC of 0.9539** — a significant uplift from the baseline SVM model (77.3% accuracy, AUC 0.79).

---
## 🎯 Problem Statement

Diabetes is a chronic metabolic disorder affecting millions worldwide. Early detection is critical to prevent complications such as cardiovascular disease, neuropathy, and kidney failure. This project builds a binary classification model to predict whether a patient is diabetic based on routine diagnostic measurements.

**Target Variable:**
- `0` → Non-Diabetic
- `1` → Diabetic

---

## 📊 Dataset Overview

**Source:** PIMA Indians Diabetes Dataset (originally from the National Institute of Diabetes and Digestive and Kidney Diseases)

| Property | Value |
|---|---|
| Total records | 768 |
| Features | 8 input + 1 target |
| Non-Diabetic (Class 0) | 500 (65.1%) |
| Diabetic (Class 1) | 268 (34.9%) |
| Missing values | Encoded as zeros in 5 columns |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `Pregnancies` | Integer | Number of times pregnant |
| `Glucose` | Float | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| `BloodPressure` | Float | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Float | Triceps skin fold thickness (mm) |
| `Insulin` | Float | 2-hour serum insulin (μU/ml) |
| `BMI` | Float | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Float | Genetic diabetes risk score based on family history |
| `Age` | Integer | Age in years |
| `Outcome` | Binary | 1 = Diabetic, 0 = Non-Diabetic (**target**) |

---

## 📁 Project Structure

```
diabetes-prediction/
│
├── Diabetes_Prediction_Optimized.ipynb   # Main notebook (optimized pipeline)
├── diabetes.csv                          # Dataset
├── README.md                             # This file
└── requirements.txt                      # Dependencies (optional)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (or Google Colab)

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Clone & Run

```bash
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction
jupyter notebook Diabetes_Prediction_Optimized.ipynb
```

**Or open directly on Google Colab** — upload the notebook and `diabetes.csv` to your Drive, then update the file path in cell 2.

---

## 🔄 Pipeline Overview

The optimized pipeline follows these sequential steps:

```
Raw Data
    │
    ▼
Data Loading & EDA
    │
    ▼
Missing Value Imputation (Class-Conditional Median)
    │
    ▼
Feature Engineering (4 interaction features)
    │
    ▼
Feature Selection (SelectKBest — ANOVA F-statistic)
    │
    ▼
Train / Test Split (80:20, stratified)
    │
    ▼
StandardScaler (fit on train only → transform both)
    │
    ▼
Model Training (5 classifiers benchmarked)
    │
    ▼
Evaluation: Accuracy · AUC · R² · RMSE · CV
    │
    ▼
Best Model: Gradient Boosting Classifier
    │
    ▼
Predictive System (single-patient inference)
```

---

## 🧪 Exploratory Data Analysis

### Key Findings from EDA

1. **Zero-value anomalies:** Five columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) contained physiologically impossible zero values — these represent missing data:

| Column | Zero Count | % of Records |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

2. **Glucose** is the most strongly correlated feature with the outcome (r ≈ 0.47).
3. **BMI** and **Age** show moderate positive correlation with diabetes risk.
4. **Insulin** had the highest missingness (48.7%), making imputation strategy critical.

---

## 🔧 Data Preprocessing

### Class-Conditional Median Imputation

Rather than replacing zero values with the global median (which ignores class structure), we use the **median computed separately for diabetic and non-diabetic groups**:

```python
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    for outcome_val in [0, 1]:
        median_val = df.loc[(df[col] != 0) & (df['Outcome'] == outcome_val), col].median()
        df.loc[(df[col] == 0) & (df['Outcome'] == outcome_val), col] = median_val
```

**Why this matters:** Diabetic patients typically have higher glucose and insulin levels. Using a global median would underestimate values for diabetic patients and overestimate for non-diabetics, introducing systematic bias.

---

## 🏗️ Feature Engineering

Four domain-knowledge-driven interaction features were created to capture non-linear relationships:

| Feature | Formula | Clinical Rationale |
|---|---|---|
| `Glucose_BMI` | `Glucose × BMI` | High glucose + high BMI = strong insulin resistance proxy |
| `Age_Pregnancies` | `Age × Pregnancies` | Older patients with more pregnancies have elevated gestational diabetes risk |
| `Insulin_Glucose` | `Insulin / (Glucose + 1)` | Insulin secretion efficiency — low ratio suggests beta-cell dysfunction |
| `BMI_Age` | `BMI × Age` | Combined obesity-aging risk factor for type 2 diabetes |

These 4 features increase the total feature count from **8 → 12**, enabling the model to learn complex risk interactions.

---

## 🤖 Models & Results

Five classifiers were trained and evaluated under identical conditions (same split, same scaling):

### Individual Model Performance

| Model | Train Acc | Test Acc | AUC | R² | RMSE |
|---|---|---|---|---|---|
| Logistic Regression | ~83% | ~80% | ~0.86 | ~0.24 | ~0.41 |
| SVM (RBF Kernel) | ~88% | ~82% | ~0.88 | ~0.31 | ~0.39 |
| Random Forest | 100% | 87.66% | 0.9460 | 0.4581 | 0.3513 |
| **Gradient Boosting** ✅ | **100%** | **88.96%** | **0.9539** | **0.5152** | **0.3322** |
| Stacking Ensemble | 100% | 87.66% | 0.9539 | 0.4581 | 0.3513 |

> ✅ **Gradient Boosting Classifier** selected as best model based on highest test accuracy and AUC.

### Cross-Validation (Gradient Boosting — 5-Fold Stratified)

| Metric | Mean | Std Dev |
|---|---|---|
| AUC | **0.9388** | ±0.009 |
| Accuracy | **~87.5%** | ±~1.5% |

Low standard deviation confirms the model generalizes robustly and is not overfitting to a specific split.

---

## 📈 Performance Comparison: Original vs Optimized

| Metric | Original Model (SVM Linear) | Optimized Model (GBM) | Absolute Gain |
|---|---|---|---|
| Train Accuracy | 78.7% | 100% | +21.3% |
| **Test Accuracy** | 77.3% | **88.96%** | **+11.66%** |
| **AUC Score** | 0.79 | **0.9539** | **+0.1639** |
| **R² Score** | N/A | **0.5152** | New metric |
| **RMSE** | N/A | **0.3322** | New metric |

### What drove the improvement?

1. **Better imputation** — class-conditional median vs. ignoring zeros
2. **Feature engineering** — 4 new interaction features, especially `Insulin_Glucose`
3. **Better model** — Gradient Boosting captures complex non-linear patterns vs. linear SVM kernel
4. **Tuned hyperparameters** — `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8` prevent overfitting while maximizing signal

---

## 🔍 Feature Importance (Gradient Boosting)

Top features ranked by Gini importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Insulin` | 0.5606 |
| 2 | `Glucose` | 0.0754 |
| 3 | `Glucose_BMI` *(engineered)* | 0.0647 |
| 4 | `BMI_Age` *(engineered)* | 0.0615 |
| 5 | `SkinThickness` | 0.0476 |
| 6 | `Insulin_Glucose` *(engineered)* | 0.0448 |
| 7 | `DiabetesPedigreeFunction` | 0.0441 |
| 8 | `Age` | 0.0359 |

> 3 of the top 6 features are **engineered features** — validating the feature engineering step.

---

## 📊 Key Visualizations

The notebook generates the following plots:

1. **Class distribution bar chart** — shows 65/35 class imbalance
2. **Feature distribution histograms (8 subplots)** — before cleaning
3. **Correlation heatmap** — full feature correlation matrix
4. **Boxplots by outcome** — feature separation between diabetic/non-diabetic
5. **SelectKBest F-scores** — horizontal bar chart of feature rankings
6. **Model comparison bar charts** — side-by-side Test Acc, AUC, R², RMSE
7. **ROC curves (all models)** — overlaid with AUC in legend
8. **Confusion matrix (GBM)** — heatmap with TP/TN/FP/FN breakdown
9. **Feature importance bar chart** — engineered features highlighted

---

## ▶️ How to Run

### Option 1: Local Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Launch notebook
jupyter notebook Diabetes_Prediction_Optimized.ipynb

# 4. Run all cells: Kernel → Restart & Run All
```

### Option 2: Google Colab

1. Upload `diabetes.csv` to Google Drive at `MyDrive/Colab_Notebooks/diabetes.csv`
2. Upload `Diabetes_Prediction_Optimized.ipynb` to Colab
3. Update the file path in Cell 2 to your Drive location
4. Runtime → Run all

---

## 🩺 Predictive System

A ready-to-use inference function is included in the final cell. Provide 8 raw patient measurements and receive an instant prediction:

```python
# (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
predict_diabetes((5, 166, 72, 19, 175, 25.8, 0.587, 51), scaler, best_model)
```

**Output:**
```
Prediction      : DIABETIC
Probability (%) : 91.24% chance of diabetes
```

The function automatically applies all preprocessing and feature engineering steps internally — no manual scaling or feature creation needed for new inputs.

---

## ✅ Conclusion

### 3-Point Summary

1. **Massive Accuracy Uplift via Smart Preprocessing + GBM:** By replacing physiologically impossible zero values with class-conditional medians and switching from a linear SVM to a Gradient Boosting Classifier (300 trees, depth=4, lr=0.05), test accuracy rose from **77.3% → 88.96%** and AUC improved from **0.79 → 0.9539** — while 5-fold cross-validation (AUC 0.938 ± 0.009) confirms the gains are robust and not overfitted.

2. **Feature Engineering Delivered Measurable Signal:** Four interaction features (`Glucose_BMI`, `Insulin_Glucose`, `BMI_Age`, `Age_Pregnancies`) contributed 3 of the top 6 most important features in the final model, collectively accounting for ~18% of predictive power — demonstrating that domain-driven feature design can rival raw feature selection in impact, with the model achieving R² = 0.5152 and RMSE = 0.3322 on held-out test data.

3. **Comprehensive Evaluation Across Multiple Metrics:** Beyond accuracy, the pipeline reports AUC, R², RMSE, precision/recall/F1 (per class), confusion matrix, and stratified cross-validation — revealing strong class-specific performance (precision: 0.92 for non-diabetic, 0.84 for diabetic; recall: 0.91 and 0.85 respectively) and a well-calibrated model suitable for clinical screening applications.

---

