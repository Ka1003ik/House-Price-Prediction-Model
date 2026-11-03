from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make the custom preprocessing class available to joblib.
from src.preprocessing import NeighborhoodMedianImputer

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("             HOUSE PRICE MODEL EVALUATION")
print("=" * 70)


# ============================================================
# 2. LOAD BASELINE AND TUNED PREDICTIONS
# ============================================================

print("\n[1] LOADING MODEL PREDICTIONS")
print("-" * 70)

baseline_path = OUTPUT_DIR / "validation_predictions.csv"
tuned_path = OUTPUT_DIR / "tuned_validation_predictions.csv"

baseline = pd.read_csv(baseline_path)
tuned = pd.read_csv(tuned_path)

actual = baseline["Actual_SalePrice"].to_numpy()

# Confirm both files use the same validation observations.
if not np.allclose(
    actual,
    tuned["Actual_SalePrice"].to_numpy()
):
    raise ValueError(
        "Baseline and tuned predictions use different validation targets."
    )

print(f"Validation records: {len(actual)}")
print("Baseline predictions loaded.")
print("Tuned predictions loaded.")
print("Validation targets match.")


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

print("\n[2] MODEL PERFORMANCE COMPARISON")
print("-" * 70)

prediction_columns = {
    "Linear Regression (Baseline)":
        (baseline, "linear_regression_prediction"),

    "Decision Tree (Baseline)":
        (baseline, "decision_tree_regression_prediction"),

    "Random Forest (Baseline)":
        (baseline, "random_forest_regression_prediction"),

    "Decision Tree (Tuned)":
        (tuned, "decision_tree_regression_prediction"),

    "Random Forest (Tuned)":
        (tuned, "random_forest_regression_prediction")
}

results = []

for model_name, (data, column) in prediction_columns.items():

    predicted = data[column].to_numpy()

    mse = mean_squared_error(actual, predicted)

    results.append({
        "Model": model_name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)


# ============================================================
# 4. SELECT CURRENT BEST MODEL
# ============================================================

print("\n[3] CURRENT BEST MODEL")
print("-" * 70)

best_result = results_df.iloc[0]

print(f"Model: {best_result['Model']}")
print(f"RMSE: {best_result['RMSE']:,.2f}")
print(f"MAE: {best_result['MAE']:,.2f}")
print(f"R2: {best_result['R2']:.4f}")


# ============================================================
# 5. ACTUAL VS PREDICTED ANALYSIS
# ============================================================

print("\n[4] ACTUAL VS PREDICTED ANALYSIS")
print("-" * 70)

best_rf = tuned["random_forest_regression_prediction"].to_numpy()

plt.figure(figsize=(9, 6))

plt.scatter(
    actual,
    best_rf,
    alpha=0.65
)

minimum = min(actual.min(), best_rf.min())
maximum = max(actual.max(), best_rf.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("Actual vs predicted chart saved.")


# ============================================================
# 6. RESIDUAL ANALYSIS
# ============================================================

print("\n[5] RESIDUAL ANALYSIS")
print("-" * 70)

# Residual = Actual value - Predicted value
residuals = actual - best_rf

print(f"Average residual: {residuals.mean():,.2f}")
print(f"Residual standard deviation: {residuals.std():,.2f}")
print(f"Minimum residual: {residuals.min():,.2f}")
print(f"Maximum residual: {residuals.max():,.2f}")

# Residuals versus predicted values
plt.figure(figsize=(9, 6))

plt.scatter(
    best_rf,
    residuals,
    alpha=0.65
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residual")
plt.title("Random Forest: Residuals vs Predicted Prices")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residuals_vs_predicted.png",
    dpi=300
)

plt.close()

print("Residual plot saved.")


# ============================================================
# 7. RESIDUAL DISTRIBUTION
# ============================================================

print("\n[6] RESIDUAL DISTRIBUTION")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.hist(
    residuals,
    bins=30,
    edgecolor="black"
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Random Forest: Residual Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300
)

plt.close()

print("Residual distribution chart saved.")


# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n[7] RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)

model_path = (
    MODEL_DIR / "random_forest_regression_tuned_pipeline.joblib"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Tuned Random Forest model not found: {model_path}"
    )

rf_pipeline = joblib.load(model_path)

# Extract the fitted Random Forest model.
rf_model = rf_pipeline.named_steps["model"]

# Extract the feature names generated by preprocessing.
fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]

feature_names = (
    fitted_preprocessor
    .named_steps["column_preprocessor"]
    .get_feature_names_out()
)

importances = rf_model.feature_importances_

if len(feature_names) != len(importances):
    raise ValueError(
        "Feature names and importance counts do not match."
    )

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nTOP 20 IMPORTANT FEATURES")
print(importance_df.head(20).round(5).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False
)

top_features = importance_df.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest - Top 20 Feature Importances")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature importance chart saved.")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("                MODEL EVALUATION COMPLETED")
print("=" * 70)

print(f"Best model by validation RMSE: {best_result['Model']}")
print(f"Validation RMSE: {best_result['RMSE']:,.2f}")
print(f"Validation MAE: {best_result['MAE']:,.2f}")
print(f"Validation R2: {best_result['R2']:.4f}")

print("\nGenerated evaluation files:")
print("1. final_model_comparison.csv")
print("2. actual_vs_predicted.png")
print("3. residuals_vs_predicted.png")
print("4. residual_distribution.png")
print("5. random_forest_feature_importance.csv")
print("6. random_forest_feature_importance.png")

print(f"\nOutput directory: {OUTPUT_DIR}")

print("\nAll evaluation tasks completed successfully!")
print("=" * 70)
