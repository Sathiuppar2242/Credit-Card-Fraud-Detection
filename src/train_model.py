import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 50


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading training data...")

train = pd.read_csv("data/train.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", train.shape)


# ============================================================
# 2. CONVERT TRANSACTION DATE/TIME
# ============================================================

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)


# ============================================================
# 3. CREATE TIME FEATURES
# ============================================================

train["transaction_hour"] = (
    train["trans_date_trans_time"].dt.hour
)

train["transaction_day"] = (
    train["trans_date_trans_time"].dt.day
)

train["transaction_month"] = (
    train["trans_date_trans_time"].dt.month
)

train["transaction_day_of_week"] = (
    train["trans_date_trans_time"].dt.dayofweek
)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "amt",
    "category",
    "gender",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_day_of_week"
]

target = "is_fraud"

X = train[features]
y = train[target]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nSelected features:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 5. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "category",
    "gender"
]

numerical_features = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_day_of_week"
]


# ============================================================
# 6. CREATE PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. TRANSFORM FEATURES
# ============================================================

X_processed = preprocessor.fit_transform(X)

print("\nOriginal feature shape:", X.shape)
print("Processed feature shape:", X_processed.shape)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X_processed,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_valid.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nValidation target distribution:")
print(y_valid.value_counts())


# ============================================================
# 9. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 11. MAKE VALIDATION PREDICTIONS
# ============================================================

y_pred = model.predict(X_valid)

y_prob = model.predict_proba(X_valid)[:, 1]


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_valid,
        y_pred
    )
)


print("\n--- Confusion Matrix ---")

print(
    confusion_matrix(
        y_valid,
        y_pred
    )
)


print("\n--- ROC-AUC Score ---")

roc_auc = roc_auc_score(
    y_valid,
    y_prob
)

print(roc_auc)


print("\n--- Average Precision Score ---")

average_precision = average_precision_score(
    y_valid,
    y_prob
)

print(average_precision)


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print("\nModel saved to: models/fraud_model.pkl")


# ============================================================
# 14. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print(
    "Preprocessor saved to: "
    "models/preprocessor.pkl"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("Credit Card Fraud Detection Model Ready")
print("========================================")