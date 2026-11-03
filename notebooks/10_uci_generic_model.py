
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generic_pipeline import (
    prepare_features_target,
    detect_feature_types,
    build_generic_pipeline
)

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "uci_real_estate"
    / "Real estate valuation data set.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Y house price of unit area"
ID_COLUMNS = ["No"]

# No categorical columns need to be forced for this dataset.
CATEGORICAL_COLUMNS = []

RANDOM_STATE = 42


print("\n" + "=" * 70)
print("          GENERIC MULTI-DATASET REGRESSION ENGINE")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] DATASET LOADING")
print("-" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_excel(
    DATA_PATH,
    sheet_name=0,
    engine="openpyxl"
)

df.columns = df.columns.astype(str).str.strip()

print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Target column: {TARGET_COLUMN}")
print(f"Identifier columns: {ID_COLUMNS}")


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

print("\n[2] FEATURE AND TARGET PREPARATION")
print("-" * 70)

X, y = prepare_features_target(
    df,
    target_column=TARGET_COLUMN,
    id_columns=ID_COLUMNS,
    categorical_columns=CATEGORICAL_COLUMNS
)

print(f"Input features: {X.shape[1]}")
print(f"Target records: {len(y)}")

print("\nInput feature names:")
for column in X.columns:
    print(f"- {column}")


# ============================================================
# 4. AUTOMATIC FEATURE DETECTION
# ============================================================

print("\n[3] AUTOMATIC FEATURE DETECTION")
print("-" * 70)

numeric_features, categorical_features = (
    detect_feature_types(
        X,
        categorical_columns=CATEGORICAL_COLUMNS
    )
)

print(f"Numerical features: {len(numeric_features)}")

for feature in numeric_features:
    print(f"  - {feature}")

print(f"\nCategorical features: {len(categorical_features)}")

if categorical_features:
    for feature in categorical_features:
        print(f"  - {feature}")
else:
    print("  None detected in this dataset.")


# ============================================================
# 5. TRAINING AND VALIDATION SPLIT
# ============================================================

print("\n[4] TRAINING AND VALIDATION SPLIT")
print("-" * 70)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_valid)}")
print("Validation percentage: 20%")


# ============================================================
# 6. DEFINE REGRESSION MODELS
# ============================================================

model_configurations = {
    "Linear Regression": {
        "model_name": "linear_regression",
        "model_params": {}
    },

    "Decision Tree Regression": {
        "model_name": "decision_tree",
        "model_params": {}
    },

    "Random Forest Regression": {
        "model_name": "random_forest",
        "model_params": {}
    }
}


# ============================================================
# 7. CROSS-VALIDATION CONFIGURATION
# ============================================================

print("\n[5] FIVE-FOLD CROSS-VALIDATION")
print("-" * 70)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("Cross-validation folds: 5")
print("Random state: 42")
print("Preprocessing will be fitted separately inside each fold.")


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n[6] MODEL TRAINING AND EVALUATION")
print("-" * 70)

results = []
trained_pipelines = {}
predictions = {}

for model_display_name, configuration in (
    model_configurations.items()
):

    print(f"\nTraining: {model_display_name}")

    pipeline = build_generic_pipeline(
        X_train,
        model_name=configuration["model_name"],
        categorical_columns=CATEGORICAL_COLUMNS,
        model_params=configuration["model_params"],
        random_state=RANDOM_STATE
    )

    # Cross-validation on training data.
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1
    )

    cv_rmse = -cv_scores

    # Fit the complete pipeline on training data.
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    validation_predictions = pipeline.predict(X_valid)

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_predictions
        )
    )

    validation_mse = mean_squared_error(
        y_valid,
        validation_predictions
    )

    validation_rmse = np.sqrt(validation_mse)

    validation_mae = mean_absolute_error(
        y_valid,
        validation_predictions
    )

    validation_r2 = r2_score(
        y_valid,
        validation_predictions
    )

    results.append({
        "Model": model_display_name,
        "CV_RMSE_Mean": cv_rmse.mean(),
        "CV_RMSE_Std": cv_rmse.std(),
        "Train_RMSE": train_rmse,
        "Validation_RMSE": validation_rmse,
        "Validation_MAE": validation_mae,
        "Validation_R2": validation_r2
    })

    trained_pipelines[model_display_name] = pipeline
    predictions[model_display_name] = validation_predictions

    print(f"Mean CV RMSE: {cv_rmse.mean():,.4f}")
    print(f"Validation RMSE: {validation_rmse:,.4f}")
    print(f"Validation R2: {validation_r2:.4f}")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n[7] GENERIC MODEL COMPARISON")
print("-" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Validation_RMSE",
    ascending=True
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "uci_model_comparison.csv",
    index=False
)


# ============================================================
# 10. SAVE TRAINED PIPELINES
# ============================================================

print("\n[8] SAVING TRAINED PIPELINES")
print("-" * 70)

for model_name, pipeline in trained_pipelines.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_generic_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )

    print(f"Saved: {filename}")


# ============================================================
# 11. SAVE VALIDATION PREDICTIONS
# ============================================================

print("\n[9] SAVING VALIDATION PREDICTIONS")
print("-" * 70)

prediction_data = {
    "Actual_Price_Per_Unit_Area": y_valid.reset_index(drop=True)
}

for model_name, model_predictions in predictions.items():

    column_name = (
        model_name.lower()
        .replace(" ", "_")
        + "_prediction"
    )

    prediction_data[column_name] = model_predictions

prediction_df = pd.DataFrame(prediction_data)

prediction_df.to_csv(
    OUTPUT_DIR / "uci_validation_predictions.csv",
    index=False
)

print("Validation predictions saved.")


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n[10] RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)

rf_name = "Random Forest Regression"
rf_pipeline = trained_pipelines[rf_name]

rf_model = rf_pipeline.named_steps["model"]

feature_names = (
    rf_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

if hasattr(rf_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print(importance_df.round(5).to_string(index=False))

    importance_df.to_csv(
        OUTPUT_DIR / "uci_feature_importance.csv",
        index=False
    )

print("Feature importance report saved.")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("          GENERIC PIPELINE EXECUTION COMPLETED")
print("=" * 70)

print(f"Dataset records: {len(df)}")
print(f"Input features: {X.shape[1]}")
print(f"Models evaluated: {len(trained_pipelines)}")

print(f"\nBest model: {best_model['Model']}")
print(f"Validation RMSE: {best_model['Validation_RMSE']:,.4f}")
print(f"Validation MAE: {best_model['Validation_MAE']:,.4f}")
print(f"Validation R2: {best_model['Validation_R2']:.4f}")

print("\nGenerated files:")
print("1. outputs/uci_model_comparison.csv")
print("2. outputs/uci_validation_predictions.csv")
print("3. outputs/uci_feature_importance.csv")

print("\nSaved pipelines:")
print("1. models/linear_regression_generic_pipeline.joblib")
print("2. models/decision_tree_regression_generic_pipeline.joblib")
print("3. models/random_forest_regression_generic_pipeline.joblib")

print("\nAll generic pipeline operations completed!")
print("=" * 70)
