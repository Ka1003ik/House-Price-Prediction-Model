
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom preprocessing components before loading joblib files.
from src.preprocessing import (
    load_ames_dataset,
    separate_features_target,
    NeighborhoodMedianImputer
)

from src.prediction import (
    predict_house_prices,
    create_competition_submission
)


DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

TUNED_MODEL_PATH = (
    MODEL_DIR / "random_forest_regression_tuned_pipeline.joblib"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "final_house_price_pipeline.joblib"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "=" * 70)
print("       FINAL HOUSE PRICE MODEL AND TEST PREDICTION")
print("=" * 70)


# ============================================================
# 2. LOAD FULL LABELLED TRAINING DATASET
# ============================================================

print("\n[1] LOADING FULL TRAINING DATASET")
print("-" * 70)

train_df = load_ames_dataset(TRAIN_PATH)

X_full, y_full = separate_features_target(
    train_df,
    target_column="SalePrice",
    id_columns=("Id",)
)

print(f"Training records: {len(X_full)}")
print(f"Input features: {X_full.shape[1]}")
print("Training dataset loaded successfully.")


# ============================================================
# 3. LOAD THE SELECTED TUNED RANDOM FOREST PIPELINE
# ============================================================

print("\n[2] LOADING SELECTED MODEL")
print("-" * 70)

if not TUNED_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Tuned model not found: {TUNED_MODEL_PATH}"
    )

selected_pipeline = joblib.load(TUNED_MODEL_PATH)

print("Tuned Random Forest pipeline loaded.")

if "model" not in selected_pipeline.named_steps:
    raise ValueError(
        "Loaded pipeline does not contain the expected model step."
    )

print(
    "Selected algorithm:",
    type(selected_pipeline.named_steps["model"]).__name__
)


# ============================================================
# 4. REFIT MODEL USING ALL LABELLED TRAINING RECORDS
# ============================================================

print("\n[3] FINAL MODEL TRAINING")
print("-" * 70)

# Clone retains the selected hyperparameters but removes
# the previous fitted state.
final_pipeline = clone(selected_pipeline)

# Fit preprocessing and Random Forest using all 1,460 rows.
final_pipeline.fit(X_full, y_full)

print("Final pipeline fitted successfully.")
print(f"Records used: {len(X_full)}")


# ============================================================
# 5. SAVE FINAL FITTED PIPELINE
# ============================================================

print("\n[4] SAVING FINAL MODEL")
print("-" * 70)

joblib.dump(
    final_pipeline,
    FINAL_MODEL_PATH
)

print(f"Final model saved: {FINAL_MODEL_PATH.name}")


# ============================================================
# 6. LOAD UNLABELLED TEST DATASET
# ============================================================

print("\n[5] LOADING KAGGLE TEST DATASET")
print("-" * 70)

test_df = load_ames_dataset(TEST_PATH)

if "Id" not in test_df.columns:
    raise ValueError("Test dataset does not contain an Id column.")

test_ids = test_df["Id"].copy()

print(f"Test records: {len(test_df)}")
print(f"Test columns: {len(test_df.columns)}")

# This dataset has no SalePrice column.
print("SalePrice available in test data:", "SalePrice" in test_df.columns)


# ============================================================
# 7. GENERATE TEST PREDICTIONS
# ============================================================

print("\n[6] GENERATING HOUSE PRICE PREDICTIONS")
print("-" * 70)

predictions = predict_house_prices(
    final_pipeline,
    test_df,
    id_column="Id"
)

print("Predictions generated successfully.")
print(f"Number of predictions: {len(predictions)}")
print(f"Minimum predicted price: {predictions.min():,.2f}")
print(f"Maximum predicted price: {predictions.max():,.2f}")
print(f"Average predicted price: {predictions.mean():,.2f}")
print(f"Median predicted price: {np.median(predictions):,.2f}")


# ============================================================
# 8. SAVE GENERAL TEST PREDICTIONS
# ============================================================

print("\n[7] SAVING TEST PREDICTIONS")
print("-" * 70)

prediction_output = pd.DataFrame({
    "Id": test_ids.to_numpy(),
    "Predicted_SalePrice": predictions
})

prediction_path = (
    OUTPUT_DIR / "test_house_price_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)

print(f"Predictions saved: {prediction_path.name}")


# ============================================================
# 9. CREATE KAGGLE SUBMISSION FILE
# ============================================================

print("\n[8] CREATING KAGGLE SUBMISSION")
print("-" * 70)

submission = create_competition_submission(
    ids=test_ids,
    predictions=predictions,
    sample_submission_path=SAMPLE_PATH
)

submission_path = (
    OUTPUT_DIR / "house_price_submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print(f"Submission file created: {submission_path.name}")
print(f"Submission records: {len(submission)}")

print("\nSubmission preview:")
print(submission.head().to_string(index=False))


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("          FINAL HOUSE PRICE PREDICTION COMPLETED")
print("=" * 70)

print(f"Final training records: {len(X_full)}")
print(f"Test records predicted: {len(predictions)}")
print(f"Final model: {FINAL_MODEL_PATH.name}")

print("\nGenerated files:")
print("1. models/final_house_price_pipeline.joblib")
print("2. outputs/test_house_price_predictions.csv")
print("3. outputs/house_price_submission.csv")

print("\nNote:")
print("Test predictions have no ground-truth labels.")
print("Therefore, test-set accuracy cannot be calculated here.")

print("\nAll final prediction operations completed successfully!")
print("=" * 70)
