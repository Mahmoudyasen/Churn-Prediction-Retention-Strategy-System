Customer Churn Prediction & Retention System

A practical, end-to-end ML pipeline that predicts which customers are likely to churn and helps you prioritize retention actions.
This notebook is organized into 11 cells (kernels) so you can run and debug step-by-step.

🚀 Quick Start

1) Environment
    # If you use Anaconda
    conda create -n churn python=3.11 -y
    conda activate churn
    pip install -U pip

    # Core dependencies
    pip install pandas numpy scikit-learn xgboost matplotlib

    # Optional (silence the pandas warning)
    pip install -U bottleneck

2) Dataset

    # Place your CSV and update the path in Kernel 2:
    CSV_PATH = r"C:\Users\ASUS\Downloads\Bank Customer Churn Prediction.csv"

3) Run
    # Open Jupyter / VS Code notebook and execute kernels in order (1 → 11).

📁 Notebook Cell-by-Cell Guide
✅ KERNEL 1 — Imports

What it does: Brings in all libraries: pandas/numpy for data, scikit-learn for pipelines/metrics, XGBoost for a strong tree model, matplotlib for plots.
Why it matters: Keeps the project reproducible and explicit—if this cell fails, fix your environment before moving on.
You can tweak: None (unless you add libraries).

✅ KERNEL 2 — Load Dataset & Fix Missing Values

What it does:

Loads the CSV into a DataFrame.

Uses ffill() to forward-fill missing values (safer than the deprecated fillna(method="ffill")).

Shows head() to sanity-check columns.

Why it matters: Bad or missing data breaks training and skews metrics. Early inspection avoids downstream surprises.
You can tweak: Replace ffill() with domain-aware imputation or drop rows/columns as needed.

✅ KERNEL 3 — Detect Target Column

What it does:

Auto-detects a churn/exit label (tries ["Churn", "Exited", "Attrition_Flag", "churn", "churn_label"]).

Splits features/target.

Converts string targets (e.g., "Yes"/"No") to 1/0.

Why it matters: Prevents common KeyError: 'Churn' not found errors when datasets use different target names.
You can tweak: Add your label name to TARGET_CANDIDATES if needed.

✅ KERNEL 4 — Identify Numeric & Categorical Columns

What it does:

Separates numeric vs. non-numeric columns.

Why it matters: Different data types require different preprocessing (scaling vs. encoding).
You can tweak: Manually adjust lists if detection isn’t perfect (e.g., IDs you want to drop, or numeric codes you want treated as categorical).

✅ KERNEL 5 — Preprocessing Setup

What it does:

Builds a ColumnTransformer that:

Scales numeric columns (StandardScaler).

One-hot encodes categoricals (OneHotEncoder(handle_unknown="ignore")).

Why it matters:

Puts preprocessing inside the pipeline so train/inference are identical.

handle_unknown="ignore" prevents crashes when new categories appear at inference.

You can tweak:

For tree-only pipelines you could skip scaling numeric features.

Use minmax scaling if your downstream models prefer it.

✅ KERNEL 6 — Define ML Models (Pipelines)

What it does:

Creates three pipelines: Logistic Regression (baseline), Random Forest, and XGBoost—each with the same preprocessing step.

Why it matters:

Pipelines avoid data leakage and keep the workflow clean.

You can compare linear vs. tree models fairly.

You can tweak:

Hyperparameters (n_estimators, max_depth, learning_rate, etc.).

Add more models (e.g., LightGBM, CatBoost).

✅ KERNEL 7 — Train/Test Split

What it does:

Splits data into train/test with stratify=y to preserve class ratio.

Why it matters:

Class imbalance is common in churn. Stratification yields more reliable evaluation.
You can tweak:

Test size (0.2 or 0.3).

Consider time-based split if your data is temporal (train on past, test on future).

✅ KERNEL 8 — Evaluation Helper

What it does:

Fits a model, predicts labels, and—when available—uses predict_proba for ROC-AUC.

Prints: precision/recall/F1, ROC-AUC, and a confusion matrix.

Why it matters:

ROC-AUC with probabilities is more informative than accuracy for imbalanced data.

You see false positives/negatives explicitly in the confusion matrix.

You can tweak:

Add average_precision_score (PR-AUC) for highly imbalanced data.

Log metrics to a file or experiment tracker (MLflow/W&B).

✅ KERNEL 9 — Train & Evaluate All Models

What it does:

Trains LR, RF, XGB in one shot and prints their metrics.

Returns the fitted pipelines for later use.

Why it matters:

Fast model comparison under identical preprocessing.
You can tweak:

Keep the best model only, or run cross-validation/grid search.

✅ KERNEL 10 — Feature Importance (XGBoost)

What it does:

Reconstructs post-transform feature names (including one-hot columns).

Plots the Top-20 XGBoost importances.

Why it matters:

Helps you understand drivers of churn and justify decisions to stakeholders.
You can tweak:

Increase Top-20 to Top-50.

Export to CSV for reporting.

Note: Importances are model-specific; LR uses coefficients, RF/XGB use split gains.

✅ KERNEL 11 — Predict New Customer

What it does:

Builds a new_customer row with original (pre-encoded) features.

Runs xgb_model.predict and prints a human-friendly message.

Why it matters:

Demonstrates inference with the same preprocessing as training (thanks to the pipeline).
You can tweak:

Replace placeholders with your real feature set.

Use predict_proba and set a custom decision threshold (e.g., flag churn if p>=0.35).

# ################################################################################################ #

💾 Saving & Loading the Best Model

    import joblib
    joblib.dump(xgb_model, "churn_xgb_pipeline.pkl")

    # Later / in production:
    best = joblib.load("churn_xgb_pipeline.pkl")
    proba = best.predict_proba(new_customer)[:, 1]

🧰 Troubleshooting

    KeyError: 'Churn' not found
    Target column name differs. Add yours to TARGET_CANDIDATES in Kernel 3, or rename the column in your CSV.

    Pandas warning about bottleneck
    Install/upgrade: pip install -U bottleneck.

    xgboost install issues on Windows
    Try a specific version: pip install xgboost==2.1.0.

    Shape mismatch when predicting a new customer
    Ensure new_customer has all original feature columns used in training (before encoding). The pipeline will handle encoding/scaling.


