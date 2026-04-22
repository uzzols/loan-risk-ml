import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
    f1_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Loan_default.csv"
MODEL_PATH = BASE_DIR / "models" / "best_loan_risk_model.joblib"
OUTPUT_PATH = BASE_DIR / "outputs" / "model_metrics.txt"
COMPARISON_PATH = BASE_DIR / "outputs" / "model_comparison.csv"

TARGET_COL = "Default"
THRESHOLD = 0.40

# Load data
df = pd.read_csv(DATA_PATH)

print("Columns in dataset:")
print(df.columns.tolist())

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COL])

# Split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Drop ID column if present
if "LoanID" in X.columns:
    X = X.drop(columns=["LoanID"])

# Identify feature types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# Shared train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing for tree-based models
numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_tree, numeric_features),
        ("cat", categorical_transformer_tree, categorical_features)
    ]
)

# Preprocessing for Logistic Regression
numeric_transformer_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_lr, numeric_features),
        ("cat", categorical_transformer_lr, categorical_features)
    ]
)

# Models
random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor_tree),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

logistic_regression_model = Pipeline(steps=[
    ("preprocessor", preprocessor_lr),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

gradient_boosting_model = Pipeline(steps=[
    ("preprocessor", preprocessor_tree),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])

models = {
    "Random Forest": random_forest_model,
    "Logistic Regression": logistic_regression_model,
    "Gradient Boosting": gradient_boosting_model
}

results = []
best_model = None
best_model_name = None
best_recall = -1

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Threshold": THRESHOLD,
        "Accuracy": round(accuracy, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "F1 Score": round(f1, 4),
        "AUC": round(auc, 4),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1]),
    })

    print(f"\n{model_name} Metrics:")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    if recall > best_recall:
        best_recall = recall
        best_model = model
        best_model_name = model_name

# Save best model
joblib.dump(best_model, MODEL_PATH)

# Save comparison table
results_df = pd.DataFrame(results)
results_df.to_csv(COMPARISON_PATH, index=False)

# Save text summary
with open(OUTPUT_PATH, "w") as f:
    f.write(f"Threshold: {THRESHOLD:.2f}\n")
    f.write(f"Best Model (by Recall): {best_model_name}\n\n")
    f.write(results_df.to_string(index=False))

print("\nModel comparison complete.")
print(f"Best model by recall: {best_model_name}")
print(f"Saved best model to: {MODEL_PATH}")
print(f"Saved summary to: {OUTPUT_PATH}")
print(f"Saved comparison CSV to: {COMPARISON_PATH}")

print("\nComparison Table:")
print(results_df.to_string(index=False))