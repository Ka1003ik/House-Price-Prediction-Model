
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
    Validate and align incoming features with the schema used during model training.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    expected_columns = None

    if hasattr(pipeline, "feature_columns_"):
        expected_columns = list(pipeline.feature_columns_)

    if expected_columns is None and hasattr(pipeline, "feature_names_in_"):
        expected_columns = list(pipeline.feature_names_in_)

    if expected_columns is None and hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            if hasattr(preprocessor, "feature_names_in_"):
                expected_columns = list(preprocessor.feature_names_in_)
            elif hasattr(preprocessor, "named_steps"):
                nested_preprocessor = preprocessor.named_steps.get(
                    "column_preprocessor"
                )
                if nested_preprocessor is not None and hasattr(
                    nested_preprocessor, "feature_names_in_"
                ):
                    expected_columns = list(nested_preprocessor.feature_names_in_)

    if expected_columns is None and hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            expected_columns = []
            for _, _, transformer_columns in preprocessor.transformers_:
                expected_columns.extend(list(transformer_columns))

    if expected_columns is None:
        raise ValueError(
            "Unable to retrieve the fitted model's input feature schema."
        )

    features = data.drop(columns=[id_column], errors="ignore").copy()

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

    numeric_columns = []
    preprocessor = None

    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")

    if preprocessor is not None:
        if hasattr(preprocessor, "transformers_"):
            for name, _, transformer_columns in preprocessor.transformers_:
                if name == "numerical":
                    numeric_columns.extend(list(transformer_columns))
        elif hasattr(preprocessor, "named_steps"):
            nested_preprocessor = preprocessor.named_steps.get(
                "column_preprocessor"
            )
            if nested_preprocessor is not None and hasattr(
                nested_preprocessor, "transformers_"
            ):
                for name, _, transformer_columns in nested_preprocessor.transformers_:
                    if name == "numerical":
                        numeric_columns.extend(list(transformer_columns))

    for column in numeric_columns:
        if column in features.columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce"
            )

    return features


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
