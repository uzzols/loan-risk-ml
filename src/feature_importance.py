import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Loan_default.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "feature_importance.csv"

TARGET_COL = "Default"

# Load data
df = pd.read_csv(DATA_PATH)

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

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# Preprocessing for Gradient Boosting
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Get transformed feature names
ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
categorical_feature_names = ohe.get_feature_names_out(categorical_features).tolist()

all_feature_names = numeric_features + categorical_feature_names

# Get feature importances
importances = model.named_steps["classifier"].feature_importances_

# Build DataFrame
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
})

# Sort descending
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Save to CSV
importance_df.to_csv(OUTPUT_PATH, index=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

print(f"\nSaved feature importance to: {OUTPUT_PATH}")