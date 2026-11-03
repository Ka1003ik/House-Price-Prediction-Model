
from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_ames_dataset
from src.generic_pipeline import (
    prepare_features_target,
    detect_feature_types,
    build_generic_pipeline
)

DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

CATEGORICAL_COLUMNS = ["MSSubClass"]

df = load_ames_dataset(DATA_PATH)

print("\n" + "=" * 65)
print("       GENERIC ENGINE - AMES DATASET VALIDATION")
print("=" * 65)

# 1. Prepare features and target
X, y = prepare_features_target(
    df,
    target_column="SalePrice",
    id_columns=["Id"],
    categorical_columns=CATEGORICAL_COLUMNS
)

# 2. Automatically detect feature types
numeric_features, categorical_features = detect_feature_types(X)

print("\nDataset records:", len(X))
print("Input features:", X.shape[1])
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# 3. Training and validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Define generic regression models
models = {
    "linear_regression": {},
    "decision_tree": {},
    "random_forest": {}
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# 5. Train and evaluate each model
for model_name, parameters in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = build_generic_pipeline(
        X_train,
        model_name=model_name,
        categorical_columns=CATEGORICAL_COLUMNS,
        model_params=parameters,
        random_state=42
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1
    )

    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    valid_predictions = pipeline.predict(X_valid)

    train_rmse = mean_squared_error(
        y_train,
        train_predictions
    ) ** 0.5

    valid_rmse = mean_squared_error(
        y_valid,
        valid_predictions
    ) ** 0.5

    valid_mae = mean_absolute_error(
        y_valid,
        valid_predictions
    )

    valid_r2 = r2_score(
        y_valid,
        valid_predictions
    )

    results.append({
        "Model": model_name,
        "CV_RMSE_Mean": -cv_scores.mean(),
        "CV_RMSE_Std": cv_scores.std(),
        "Train_RMSE": train_rmse,
        "Validation_RMSE": valid_rmse,
        "Validation_MAE": valid_mae,
        "Validation_R2": valid_r2
    })

    joblib.dump(
        pipeline,
        MODEL_DIR / f"ames_generic_{model_name}.joblib"
    )

    print(f"CV RMSE: {-cv_scores.mean():,.2f}")
    print(f"Validation RMSE: {valid_rmse:,.2f}")
    print(f"Validation R2: {valid_r2:.4f}")

# 6. Save results
results_df = pd.DataFrame(results).sort_values(
    by="Validation_RMSE"
)

results_df.to_csv(
    OUTPUT_DIR / "ames_generic_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("       GENERIC AMES MODEL COMPARISON")
print("=" * 65)

print(results_df.round(4).to_string(index=False))

print("\nGeneric engine successfully tested on Ames dataset!")
print("=" * 65)
