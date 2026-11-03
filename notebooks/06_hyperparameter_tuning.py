
from pathlib import Path
import sys
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_ames_dataset,
    separate_features_target
)

from src.model_tuning import tune_regression_models


DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 70)
print("          HOUSE PRICE MODEL - HYPERPARAMETER TUNING")
print("=" * 70)


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

print("\n[1] LOADING ORIGINAL DATASET")
print("-" * 70)

df = load_ames_dataset(DATA_PATH)

X, y = separate_features_target(
    df,
    target_column="SalePrice",
    id_columns=("Id",)
)

print(f"Total records: {len(df)}")
print(f"Original input features: {X.shape[1]}")
print("Dataset loaded successfully.")


# ============================================================
# 3. RECREATE THE ORIGINAL TRAIN/VALIDATION SPLIT
# ============================================================

print("\n[2] TRAINING AND VALIDATION SPLIT")
print("-" * 70)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_valid)}")

print("\nImportant:")
print("The validation dataset will remain untouched during")
print("cross-validation and hyperparameter selection.")


# ============================================================
# 4. RUN HYPERPARAMETER TUNING
# ============================================================

print("\n[3] HYPERPARAMETER OPTIMIZATION")
print("-" * 70)

results, tuned_models, tuning_details = (
    tune_regression_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
        random_state=42,
        n_iter=10,
        cv_splits=5
    )
)


# ============================================================
# 5. DISPLAY MODEL COMPARISON
# ============================================================

print("\n[4] OPTIMIZED MODEL PERFORMANCE")
print("-" * 70)

display_results = results.copy()

for column in [
    "CV_RMSE",
    "Train_RMSE",
    "Validation_RMSE",
    "Validation_MAE"
]:
    display_results[column] = display_results[column].map(
        lambda value: (
            "Not applicable"
            if pd.isna(value)
            else f"{value:,.2f}"
        )
    )

display_results["Validation_R2"] = (
    display_results["Validation_R2"].map(
        lambda value: f"{value:.4f}"
    )
)

print(display_results.to_string(index=False))


# ============================================================
# 6. DISPLAY BEST HYPERPARAMETERS
# ============================================================

print("\n[5] BEST HYPERPARAMETERS")
print("-" * 70)

for model_name, details in tuning_details.items():

    print(f"\n{model_name}")
    print("-" * 50)

    print(f"Best CV RMSE: {details['Best_CV_RMSE']:,.2f}")

    for parameter, value in details["Best_Parameters"].items():
        print(f"{parameter}: {value}")


# ============================================================
# 7. SAVE OPTIMIZED MODELS
# ============================================================

print("\n[6] SAVING OPTIMIZED MODELS")
print("-" * 70)

for model_name, pipeline in tuned_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_tuned_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )

    print(f"Saved: {filename}")


# ============================================================
# 8. SAVE COMPARISON RESULTS
# ============================================================

comparison_path = (
    OUTPUT_DIR / "tuned_model_comparison.csv"
)

results.to_csv(
    comparison_path,
    index=False
)

print(f"\nComparison saved: {comparison_path.name}")


# ============================================================
# 9. SAVE BEST PARAMETERS
# ============================================================

parameters_path = (
    OUTPUT_DIR / "best_hyperparameters.json"
)

with open(
    parameters_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        tuning_details,
        file,
        indent=4,
        default=str
    )

print(f"Hyperparameters saved: {parameters_path.name}")


# ============================================================
# 10. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_data = {
    "Actual_SalePrice": y_valid.reset_index(drop=True)
}

for model_name, pipeline in tuned_models.items():

    prediction_column = (
        model_name.lower().replace(" ", "_")
        + "_prediction"
    )

    prediction_data[prediction_column] = (
        pipeline.predict(X_valid)
    )

predictions_df = pd.DataFrame(prediction_data)

predictions_df.to_csv(
    OUTPUT_DIR / "tuned_validation_predictions.csv",
    index=False
)

print("Validation predictions saved.")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

best_result = results.iloc[0]

print("\n" + "=" * 70)
print("             HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)

print(f"\nBest model by validation RMSE: {best_result['Model']}")
print(f"Validation RMSE: {best_result['Validation_RMSE']:,.2f}")
print(f"Validation MAE: {best_result['Validation_MAE']:,.2f}")
print(f"Validation R2: {best_result['Validation_R2']:.4f}")

print(f"\nTrained pipelines saved in: {MODEL_DIR}")
print(f"Results saved in: {OUTPUT_DIR}")

print("\nAll tuning operations completed successfully!")
print("=" * 70)
