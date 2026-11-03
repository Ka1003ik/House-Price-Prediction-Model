
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import the custom class so saved pipelines can be loaded.
from src.preprocessing import NeighborhoodMedianImputer


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

def load_prediction_pipeline(model_path):
    """Load a saved, fitted house-price prediction pipeline."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# 2. PREPARE INPUT FEATURES
# ============================================================


def prepare_prediction_features(
    data,
    pipeline,
    id_column="Id"
):
    """
    Validate and align incoming features with the
    feature schema stored in the fitted preprocessing pipeline.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    # Find the original training feature names.
    # Our model contains a nested preprocessing pipeline.
    expected_columns = None

    # Option 1: Explicit schema saved with the model.
    if hasattr(pipeline, "feature_columns_"):
        expected_columns = pipeline.feature_columns_

    # Option 2: Retrieve schema from the nested ColumnTransformer.
    if expected_columns is None and hasattr(pipeline, "named_steps"):

        if "preprocessor" in pipeline.named_steps:

            fitted_preprocessor = (
                pipeline.named_steps["preprocessor"]
            )

            if (
                hasattr(fitted_preprocessor, "named_steps")
                and "column_preprocessor"
                in fitted_preprocessor.named_steps
            ):

                column_preprocessor = (
                    fitted_preprocessor
                    .named_steps["column_preprocessor"]
                )

                if hasattr(
                    column_preprocessor,
                    "feature_names_in_"
                ):
                    expected_columns = (
                        column_preprocessor.feature_names_in_
                    )

    # Option 3: Fall back to the outer pipeline schema.
    if expected_columns is None:
        if hasattr(pipeline, "feature_names_in_"):
            expected_columns = pipeline.feature_names_in_

    if expected_columns is None:
        raise ValueError(
            "Unable to retrieve the fitted model's input feature schema."
        )

    expected_columns = list(expected_columns)

    # Remove identifier column.
    features = data.drop(
        columns=[id_column],
        errors="ignore"
    ).copy()

    # Check missing and unexpected columns.
    missing_columns = [
        col for col in expected_columns
        if col not in features.columns
    ]

    unexpected_columns = [
        col for col in features.columns
        if col not in expected_columns
    ]

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Input feature schema does not match the trained model.\n"
            f"Missing columns: {missing_columns}\n"
            f"Unexpected columns: {unexpected_columns}"
        )

    features = features[expected_columns]

# ------------------------------------------------
# SAFETY CHECK: ENFORCE TRAINED NUMERICAL SCHEMA
# ------------------------------------------------

# Retrieve the fitted numerical feature list
# from the trained ColumnTransformer.
    inner_pipeline = pipeline.named_steps.get("preprocessor")

    if inner_pipeline is None:
        raise ValueError(
            "Trained pipeline does not contain a preprocessor."
        )
    column_preprocessor = (
        inner_pipeline.named_steps.get("column_preprocessor")
        )
    
    if column_preprocessor is None:
        raise ValueError(
        "ColumnTransformer not found in fitted pipeline."
        )
    numeric_columns = []

    for name, transformer, columns in column_preprocessor.transformers_:
         if name == "numerical":
            numeric_columns.extend(list(columns))

# Convert test values to numeric types.
# Unexpected text such as "NA" becomes NaN.
# The fitted median imputer handles those values.
    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
            )
    return features


    """
    Remove the identifier and align incoming features
    with the exact schema used during model training.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if not hasattr(pipeline, "feature_names_in_"):
        raise ValueError(
            "The fitted pipeline does not contain its input schema."
        )

    features = data.drop(
        columns=[id_column],
        errors="ignore"
    ).copy()

    expected_columns = list(pipeline.feature_names_in_)

    missing_columns = [
        col for col in expected_columns
        if col not in features.columns
    ]

    unexpected_columns = [
        col for col in features.columns
        if col not in expected_columns
    ]

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Input feature schema does not match the trained model.\n"
            f"Missing columns: {missing_columns}\n"
            f"Unexpected columns: {unexpected_columns}"
        )

    # Ensure columns have the same order as training.
    return features[expected_columns]


# ============================================================
# 3. GENERATE HOUSE PRICE PREDICTIONS
# ============================================================

def predict_house_prices(
    pipeline,
    data,
    id_column="Id"
):
    """Generate predictions using a fitted pipeline."""

    features = prepare_prediction_features(
        data,
        pipeline,
        id_column
    )

    predictions = np.asarray(
        pipeline.predict(features)
    ).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model generated missing or infinite predictions."
        )

    if (predictions < 0).any():
        raise ValueError(
            "Model generated negative house-price predictions."
        )

    return predictions


# ============================================================
# 4. CREATE KAGGLE COMPETITION SUBMISSION
# ============================================================

def create_competition_submission(
    ids,
    predictions,
    sample_submission_path
):
    """
    Generate a submission matching Kaggle's sample format.
    """

    ids = pd.Series(ids).reset_index(drop=True)
    predictions = np.asarray(predictions).reshape(-1)

    if len(ids) != len(predictions):
        raise ValueError(
            "Number of IDs and predictions do not match."
        )

    if ids.duplicated().any():
        raise ValueError("Duplicate test IDs detected.")

    sample_path = Path(sample_submission_path)

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample submission not found: {sample_path}"
        )

    sample = pd.read_csv(sample_path)

    required_columns = {"Id", "SalePrice"}

    if not required_columns.issubset(sample.columns):
        raise ValueError(
            "Sample submission must contain Id and SalePrice columns."
        )

    prediction_lookup = pd.Series(
        predictions,
        index=ids.to_numpy()
    )

    sample_ids = sample["Id"]

    if set(sample_ids) != set(ids):
        raise ValueError(
            "Test IDs do not match the sample submission IDs."
        )

    sample["SalePrice"] = sample_ids.map(
        prediction_lookup
    )

    if sample["SalePrice"].isna().any():
        raise ValueError(
            "Some submission prices could not be matched to test IDs."
        )

    return sample
