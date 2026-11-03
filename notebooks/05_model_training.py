
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy import sparse


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import train_and_evaluate_models


OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 65)
print("          HOUSE PRICE REGRESSION MODEL TRAINING")
print("=" * 65)


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

print("\n[1] LOADING PREPROCESSED DATA")
print("-" * 65)


def load_processed_matrix(file_stem):
    """
    Load a processed matrix saved in NPZ or NPY format.
    """

    npz_path = OUTPUT_DIR / f"{file_stem}.npz"
    npy_path = OUTPUT_DIR / f"{file_stem}.npy"

    if npz_path.exists():
        return sparse.load_npz(npz_path)

    if npy_path.exists():
        return np.load(npy_path)

    raise FileNotFoundError(
        f"Processed matrix not found: {file_stem}"
    )


X_train = load_processed_matrix("X_train_processed")
X_valid = load_processed_matrix("X_valid_processed")

y_train = pd.read_csv(
    OUTPUT_DIR / "y_train.csv"
).squeeze("columns")

y_valid = pd.read_csv(
    OUTPUT_DIR / "y_valid.csv"
).squeeze("columns")


print(f"Training records: {X_train.shape[0]}")
print(f"Validation records: {X_valid.shape[0]}")
print(f"Number of processed features: {X_train.shape[1]}")


# ============================================================
# 3. VERIFY INPUT DATA
# ============================================================

print("\n[2] INPUT DATA VALIDATION")
print("-" * 65)

if X_train.shape[0] != len(y_train):
    raise ValueError("Training feature and target counts differ.")

if X_valid.shape[0] != len(y_valid):
    raise ValueError("Validation feature and target counts differ.")

if X_train.shape[1] != X_valid.shape[1]:
    raise ValueError("Training and validation feature counts differ.")

if y_train.isna().any() or y_valid.isna().any():
    raise ValueError("Missing target values detected.")

print("Training data validation: PASSED")
print("Validation data validation: PASSED")


# ============================================================
# 4. TRAIN AND EVALUATE REGRESSION MODELS
# ============================================================

print("\n[3] MODEL TRAINING")
print("-" * 65)

results, trained_models, predictions = (
    train_and_evaluate_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
        random_state=42
    )
)


# ============================================================
# MODEL OVERFITTING ANALYSIS
# ============================================================

print("\n[4] TRAINING VS VALIDATION PERFORMANCE")
print("-" * 65)

diagnostics = []

for model_name, model in trained_models.items():

    # Predictions on training data
    train_predictions = model.predict(X_train)

    # Predictions on validation data
    valid_predictions = predictions[model_name]

    # Calculate RMSE
    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    valid_rmse = np.sqrt(
        mean_squared_error(y_valid, valid_predictions)
    )

    # Calculate R2
    train_r2 = r2_score(y_train, train_predictions)
    valid_r2 = r2_score(y_valid, valid_predictions)

    diagnostics.append({
        "Model": model_name,
        "Train_RMSE": train_rmse,
        "Validation_RMSE": valid_rmse,
        "RMSE_Gap": valid_rmse - train_rmse,
        "Train_R2": train_r2,
        "Validation_R2": valid_r2
    })

diagnostics_df = pd.DataFrame(diagnostics)

print(diagnostics_df.round(4).to_string(index=False))

print("\nInterpretation:")
print("A large training-validation performance gap")
print("may indicate overfitting.")

# ============================================================
# 5. DISPLAY MODEL COMPARISON
# ============================================================

print("\n[4] REGRESSION MODEL PERFORMANCE COMPARISON")
print("-" * 65)

display_results = results.copy()

for column in ["MSE", "RMSE", "MAE"]:
    display_results[column] = display_results[column].map(
        lambda value: f"{value:,.2f}"
    )

display_results["R2"] = display_results["R2"].map(
    lambda value: f"{value:.4f}"
)

print(display_results.to_string(index=False))

print("\nMetric interpretation:")
print("MSE  : Mean Squared Error - lower is better.")
print("RMSE : Root Mean Squared Error - lower is better.")
print("MAE  : Mean Absolute Error - lower is better.")
print("R2   : R-squared - higher is better.")


# ============================================================
# 6. SAVE TRAINED MODELS
# ============================================================

print("\n[5] SAVING TRAINED MODELS")
print("-" * 65)

for model_name, model in trained_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_model.joblib"
    )

    model_path = MODEL_DIR / filename

    joblib.dump(model, model_path)

    print(f"Saved: {filename}")


# ============================================================
# 7. SAVE MODEL COMPARISON RESULTS
# ============================================================

comparison_path = OUTPUT_DIR / "model_comparison.csv"

results.to_csv(
    comparison_path,
    index=False
)

print(f"\nModel comparison saved: {comparison_path.name}")


# ============================================================
# 8. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_data = {
    "Actual_SalePrice": np.asarray(y_valid)
}

for model_name, model_predictions in predictions.items():

    column_name = (
        model_name.lower()
        .replace(" ", "_")
        + "_prediction"
    )

    prediction_data[column_name] = model_predictions


predictions_df = pd.DataFrame(prediction_data)

predictions_path = OUTPUT_DIR / "validation_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(f"Validation predictions saved: {predictions_path.name}")


# ============================================================
# 9. IDENTIFY BEST MODEL
# ============================================================

best_model = results.iloc[0]

print("\n" + "=" * 65)
print("                 MODEL TRAINING SUMMARY")
print("=" * 65)

print(f"\nBest model by validation MSE: {best_model['Model']}")
print(f"Validation MSE: {best_model['MSE']:,.2f}")
print(f"Validation RMSE: {best_model['RMSE']:,.2f}")
print(f"Validation MAE: {best_model['MAE']:,.2f}")
print(f"Validation R2: {best_model['R2']:.4f}")

print("\nAll three regression models trained successfully.")
print(f"Trained models saved in: {MODEL_DIR}")
print("=" * 65)
