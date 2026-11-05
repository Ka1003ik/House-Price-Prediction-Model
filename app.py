import hashlib
import io
import os
import re
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.generic_pipeline import (
    build_generic_pipeline,
    prepare_features_target,
)

from src.explainability import (
    calculate_permutation_importance,
    calculate_shap_values
)

from src.prediction import (
    load_prediction_pipeline,
    predict_house_prices,
    prepare_prediction_features,
)

# Project paths are initialized before the workflow so Section 12 can
# validate a previously saved model even when no training dataset is uploaded.
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
ACTIVE_MODEL_POINTER_PATH = MODELS_ROOT / "active_model.json"


def load_active_model_location():
    """Load the last successfully trained dataset-specific model directory."""
    if not ACTIVE_MODEL_POINTER_PATH.is_file():
        return MODELS_ROOT, "legacy_root", "Legacy model folder"

    try:
        pointer = json.loads(ACTIVE_MODEL_POINTER_PATH.read_text(encoding="utf-8"))
        relative_directory = pointer.get("model_dir", ".")
        storage_key = pointer.get("storage_key", "legacy_root")
        display_name = pointer.get("dataset_name", "Saved dataset")
        candidate = (MODELS_ROOT / relative_directory).resolve()
        if not candidate.is_relative_to(MODELS_ROOT.resolve()):
            raise ValueError("Active model directory points outside the models folder.")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate, storage_key, display_name
    except Exception:
        # A damaged pointer must never make the existing legacy model inaccessible.
        return MODELS_ROOT, "legacy_root", "Legacy model folder"


def dataset_storage_location(dataset_name, dataset_signature, target_column, feature_columns):
    """Return an isolated model directory for a dataset and target."""
    # Keep the original Ames artifacts usable in their current root location.
    # The legacy directory is selected only when its metadata identifies the
    # same uploaded filename and target; new datasets never write into it.
    legacy_metadata_path = MODELS_ROOT / "best_model_metadata.json"
    if legacy_metadata_path.is_file():
        try:
            legacy_metadata = json.loads(
                legacy_metadata_path.read_text(encoding="utf-8")
            )
            if (
                legacy_metadata.get("training_dataset_name") == dataset_name
                and legacy_metadata.get("target_column") == target_column
                and legacy_metadata.get("feature_columns") == list(feature_columns)
            ):
                return MODELS_ROOT, "legacy_root"
        except Exception:
            pass

    stem = Path(dataset_name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:40] or "dataset"
    identity = f"{dataset_signature}:{target_column}".encode("utf-8")
    short_hash = hashlib.sha256(identity).hexdigest()[:12]
    storage_key = f"{slug}_{short_hash}"
    return MODELS_ROOT / "datasets" / storage_key, storage_key


def save_active_model_location(model_directory, storage_key, dataset_name):
    """Atomically remember which dataset's model should load on app restart."""
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    relative_directory = model_directory.resolve().relative_to(
        MODELS_ROOT.resolve()
    ).as_posix()
    payload = {
        "storage_key": storage_key,
        "model_dir": relative_directory or ".",
        "dataset_name": dataset_name,
    }
    temporary_path = ACTIVE_MODEL_POINTER_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    os.replace(temporary_path, ACTIVE_MODEL_POINTER_PATH)


MODEL_DIR, ACTIVE_MODEL_STORAGE_KEY, ACTIVE_MODEL_DISPLAY_NAME = (
    load_active_model_location()
)
TRAINING_ARTIFACTS_PATH = MODEL_DIR / "training_artifacts.joblib"

# ==================================================
# HELPER FUNCTIONS
# ==================================================


def load_uploaded_dataset(uploaded_file):
    """Load CSV/Excel data and preserve known Ames absence categories."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        data = pd.read_csv(
            io.BytesIO(file_bytes),
            keep_default_na=False,
            na_values=[""],
        )
    else:
        data = pd.read_excel(
            io.BytesIO(file_bytes),
            keep_default_na=False,
            na_values=[""],
        )

    # Recognize the commonly used Ames Housing training dataset by
    # several characteristic columns, not by its filename alone.
    ames_markers = {
        "SalePrice",
        "MSSubClass",
        "Neighborhood",
        "LotFrontage",
    }
    is_ames_dataset = ames_markers.issubset(set(data.columns))

    if is_ames_dataset:
        # In these columns, the literal NA means that the property
        # does not have that feature. Preserve it as an explicit category.
        absence_columns = [
            "Alley",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "PoolQC",
            "Fence",
            "MiscFeature",
        ]

        for column in absence_columns:
            if column in data.columns:
                data[column] = data[column].replace("NA", "None")

        # For MasVnrType, NA represents unknown information rather
        # than the absence category, so keep it genuinely missing.
        if "MasVnrType" in data.columns:
            data["MasVnrType"] = data["MasVnrType"].replace(
                "NA", np.nan
            )

        # These numeric fields may be read as object columns when
        # blanks are present. Convert them explicitly.
        for column in ["MasVnrArea", "GarageYrBlt", "LotFrontage"]:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column], errors="coerce"
                )
    else:
        # For general datasets, interpret common text sentinels as
        # missing values. Ames absence values were handled above.
        missing_tokens = ["NA", "N/A", "n/a", "null", "NULL", "NaN", "nan"]
        object_columns = data.select_dtypes(
            include=["object", "string"]
        ).columns
        for column in object_columns:
            data[column] = data[column].replace(
                missing_tokens, np.nan
            )

    return data, is_ames_dataset


def infer_target_display_config(target_column, dataset_name="", columns=None):
    """Infer a safe display label and unit for recognized datasets."""
    column_set = set(columns or [])

    if target_column == "SalePrice" and {
        "MSSubClass", "Neighborhood", "LotFrontage"
    }.issubset(column_set):
        return "Sale price", "USD ($)"

    uci_features = {"X1", "X2", "X3", "X4", "X5", "X6"}
    if target_column == "Y house price of unit area" and uci_features.issubset(column_set):
        return "House price per unit area", "10,000 TWD / Ping"

    readable_name = str(target_column).replace("_", " ").strip().title()
    return readable_name or "Target value", "Unspecified units"


def target_unit_label(unit):
    """Return a non-empty unit string suitable for chart/table labels."""
    cleaned = str(unit or "").strip()
    return cleaned if cleaned else "Unspecified units"


def target_metric_label(metric_name, unit, squared=False):
    """Build a metric label with the target's measurement unit."""
    unit_text = target_unit_label(unit)
    if squared:
        return f"{metric_name} ({unit_text})²"
    return f"{metric_name} ({unit_text})"


def clear_training_state():
    """Remove cached model results when the dataset/configuration changes."""
    keys_to_clear = [
        "training_results",
        "trained_models",
        "validation_actual",
        "validation_features",
        "best_model_name",
        "best_model_path",
        "feature_columns",
        "_training_config_signature",
        "_training_dataset_signature",
        "_training_dataset_name",
        "_target_display_name",
        "_target_unit",
        "actual_predicted_model",
        "feature_importance",
        "shap_values",
        "prediction_results",
        "prediction_values",
        "prediction_file_name",
        "_prediction_file_signature",
        "_prediction_results_signature",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def save_training_artifacts():
    """Persist the complete dashboard context for browser/app restarts."""
    required_keys = [
        "training_results",
        "trained_models",
        "validation_actual",
        "validation_features",
        "best_model_name",
        "best_model_path",
        "feature_columns",
    ]
    missing = [key for key in required_keys if key not in st.session_state]
    if missing:
        raise RuntimeError(
            f"Cannot save training artifacts; missing session data: {missing}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": 1,
        "training_results": st.session_state["training_results"],
        "trained_models": st.session_state["trained_models"],
        "validation_actual": np.asarray(
            st.session_state["validation_actual"], dtype=float
        ),
        "validation_features": st.session_state["validation_features"],
        "best_model_name": st.session_state["best_model_name"],
        "best_model_path": st.session_state["best_model_path"],
        "feature_columns": list(st.session_state["feature_columns"]),
        "training_config_signature": st.session_state.get(
            "_training_config_signature"
        ),
        "training_dataset_signature": st.session_state.get(
            "_training_dataset_signature"
        ),
        "training_dataset_name": st.session_state.get(
            "_training_dataset_name", "Previously trained dataset"
        ),
        "target_display_name": st.session_state.get(
            "_target_display_name", "Target value"
        ),
        "target_unit": st.session_state.get(
            "_target_unit", "Unspecified units"
        ),
        "model_storage_key": st.session_state.get("_active_model_storage_key"),
        "feature_importance": st.session_state.get("feature_importance"),
        "shap_values": st.session_state.get("shap_values"),
    }

    temporary_path = TRAINING_ARTIFACTS_PATH.with_suffix(
        TRAINING_ARTIFACTS_PATH.suffix + ".tmp"
    )
    try:
        joblib.dump(payload, temporary_path, compress=3)
        os.replace(temporary_path, TRAINING_ARTIFACTS_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

    stat = TRAINING_ARTIFACTS_PATH.stat()
    st.session_state["_training_artifact_file_signature"] = (
        stat.st_size, stat.st_mtime_ns
    )


def restore_training_artifacts():
    """Restore the most recent successful training run into Streamlit state."""
    if not TRAINING_ARTIFACTS_PATH.is_file():
        return False, None

    try:
        payload = joblib.load(TRAINING_ARTIFACTS_PATH)
        if not isinstance(payload, dict) or payload.get("artifact_version") != 1:
            raise ValueError("Unsupported or invalid training-artifact file.")

        required = {
            "training_results",
            "trained_models",
            "validation_actual",
            "validation_features",
            "best_model_name",
            "best_model_path",
            "feature_columns",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                f"Training-artifact file is incomplete: {sorted(missing)}"
            )

        results = payload["training_results"]
        models = payload["trained_models"]
        actual = np.asarray(payload["validation_actual"], dtype=float).reshape(-1)
        features = payload["validation_features"]
        feature_columns = payload["feature_columns"]

        if not isinstance(results, pd.DataFrame) or results.empty:
            raise ValueError("Saved training results are empty or invalid.")
        if not isinstance(models, dict) or not models:
            raise ValueError("Saved trained-model collection is empty or invalid.")
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("Saved validation features are empty or invalid.")
        if len(actual) != len(features) or not np.isfinite(actual).all():
            raise ValueError("Saved validation targets do not match the validation features.")
        if list(features.columns) != list(feature_columns):
            raise ValueError("Saved feature-column metadata does not match validation data.")
        if payload["best_model_name"] not in models:
            raise ValueError("Saved best model is missing from the trained-model collection.")

        st.session_state["training_results"] = results
        st.session_state["trained_models"] = models
        st.session_state["validation_actual"] = actual
        st.session_state["validation_features"] = features
        st.session_state["best_model_name"] = payload["best_model_name"]
        st.session_state["best_model_path"] = payload["best_model_path"]
        st.session_state["feature_columns"] = list(feature_columns)
        st.session_state["_training_config_signature"] = payload.get(
            "training_config_signature"
        )
        st.session_state["_training_dataset_signature"] = payload.get(
            "training_dataset_signature"
        )
        st.session_state["_training_dataset_name"] = payload.get(
            "training_dataset_name", "Previously trained dataset"
        )
        if payload.get("target_display_name"):
            st.session_state["_target_display_name"] = payload["target_display_name"]
        if payload.get("target_unit"):
            st.session_state["_target_unit"] = payload["target_unit"]
        st.session_state["_training_dataset_signature"] = payload.get(
            "training_dataset_signature"
        )

        if payload.get("feature_importance") is not None:
            st.session_state["feature_importance"] = payload["feature_importance"]
        if payload.get("shap_values") is not None:
            st.session_state["shap_values"] = payload["shap_values"]

        stat = TRAINING_ARTIFACTS_PATH.stat()
        st.session_state["_training_artifact_file_signature"] = (
            stat.st_size, stat.st_mtime_ns
        )
        return True, None

    except Exception as error:
        return False, str(error)


def rebuild_dashboard_from_existing_models(
    dataframe,
    target_column,
    test_size,
    random_state,
    saved_details,
):
    """Recreate validation dashboards from already-fitted legacy model files.

    This migration evaluates the saved pipelines on a reproducible validation
    split; it does not fit or retrain the saved model pipelines. CV metrics
    cannot be reconstructed from the old files, so those fields remain NaN.
    """
    metadata = saved_details["metadata"]
    saved_target = metadata["target_column"]
    saved_features = list(metadata["feature_columns"])
    expected_validation_percentage = float(
        metadata.get("validation_percentage", 20)
    )
    expected_random_state = int(metadata.get("random_state", 42))

    if float(test_size) != expected_validation_percentage:
        raise ValueError(
            f"Use the original validation split of {expected_validation_percentage:g}% "
            "to recover predictions from the already-fitted models."
        )
    if int(random_state) != expected_random_state:
        raise ValueError(
            f"Use the original random state {expected_random_state} "
            "to recover predictions from the already-fitted models."
        )

    if target_column != saved_target:
        raise ValueError(
            f"Select the original training target '{saved_target}' to restore this dashboard."
        )
    if saved_target not in dataframe.columns:
        raise ValueError(f"Original target column '{saved_target}' is missing from the uploaded dataset.")

    missing_features = [c for c in saved_features if c not in dataframe.columns]
    if missing_features:
        raise ValueError(
            "The uploaded training dataset does not match the saved model. "
            f"Missing features: {missing_features[:12]}"
        )

    inferred_ids = [
        c for c in dataframe.columns
        if c != saved_target and c not in saved_features
    ]

    # Recover the exact categorical columns from the fitted best pipeline.
    preprocessor = saved_details["pipeline"].named_steps.get("preprocessor")
    categorical_features = []
    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        for name, _, columns in preprocessor.transformers_:
            if name == "categorical":
                categorical_features = list(columns)
                break

    X, y = prepare_features_target(
        dataframe,
        target_column=saved_target,
        id_columns=inferred_ids,
        categorical_columns=categorical_features,
    )
    if set(X.columns) != set(saved_features):
        raise ValueError(
            "The uploaded dataset's feature schema differs from the saved model."
        )
    X = X[saved_features].copy()

    numeric_columns = X.select_dtypes(include=["number"]).columns
    if len(numeric_columns):
        X.loc[:, numeric_columns] = X.loc[:, numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=float(test_size) / 100,
        random_state=int(random_state),
    )

    model_names = [
        "linear_regression",
        "decision_tree",
        "random_forest",
    ]
    results = []
    trained_models = {}

    for model_name in model_names:
        model_path = MODEL_DIR / f"{model_name}_streamlit.joblib"
        if not model_path.is_file():
            continue

        fitted_pipeline = joblib.load(model_path)
        train_predictions = np.asarray(
            fitted_pipeline.predict(X_train), dtype=float
        ).reshape(-1)
        valid_predictions = np.asarray(
            fitted_pipeline.predict(X_valid), dtype=float
        ).reshape(-1)

        if not np.isfinite(train_predictions).all() or not np.isfinite(valid_predictions).all():
            raise ValueError(f"Saved {model_name} contains invalid predictions.")

        mse = float(mean_squared_error(y_valid, valid_predictions))
        results.append({
            "Model": model_name,
            "CV_RMSE_Mean": np.nan,
            "CV_RMSE_Std": np.nan,
            "Train_RMSE": float(np.sqrt(mean_squared_error(y_train, train_predictions))),
            "Validation_MSE": mse,
            "Validation_RMSE": float(np.sqrt(mse)),
            "Validation_MAE": float(mean_absolute_error(y_valid, valid_predictions)),
            "Validation_R2": float(r2_score(y_valid, valid_predictions)),
        })
        trained_models[model_name] = {
            "pipeline": fitted_pipeline,
            "predictions": valid_predictions,
            "model_path": str(model_path),
        }

    if not trained_models:
        raise FileNotFoundError(
            "No previously saved individual model files were found in the models folder."
        )

    results_df = pd.DataFrame(results)
    best_model_name = str(
        results_df.loc[results_df["Validation_RMSE"].idxmin(), "Model"]
    )
    best_pipeline = trained_models[best_model_name]["pipeline"]

    # Keep the persistent best-model file aligned with the recovered evaluation.
    best_model_path = MODEL_DIR / "best_house_price_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    restored_metadata = dict(metadata)
    restored_metadata.update({
        "best_model_name": best_model_name,
        "model_file": str(best_model_path.relative_to(PROJECT_ROOT)),
        "feature_columns": saved_features,
        "target_column": saved_target,
        "categorical_columns": categorical_features,
        "validation_percentage": float(test_size),
        "random_state": int(random_state),
        "dashboard_recovered_from_existing_models": True,
    })
    metadata_path = MODEL_DIR / "best_model_metadata.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    with open(temporary_metadata, "w", encoding="utf-8") as file:
        json.dump(restored_metadata, file, indent=4)
    os.replace(temporary_metadata, metadata_path)

    st.session_state["training_results"] = results_df
    st.session_state["trained_models"] = trained_models
    st.session_state["validation_actual"] = np.asarray(y_valid, dtype=float)
    st.session_state["validation_features"] = X_valid
    st.session_state["feature_columns"] = saved_features
    st.session_state["best_model_name"] = best_model_name
    st.session_state["best_model_path"] = str(best_model_path)
    st.session_state["_training_config_signature"] = None
    st.session_state["_training_dataset_signature"] = None
    st.session_state["_training_dataset_name"] = "Recovered from existing saved model files"
    st.session_state.pop("feature_importance", None)
    st.session_state.pop("shap_values", None)
    save_training_artifacts()

    return best_model_name, results_df


def get_saved_model_details():
    """Validate and return the persisted trained model and its metadata."""

    metadata_path = MODEL_DIR / "best_model_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError("Saved model metadata was not found.")

    metadata_bytes = metadata_path.read_bytes()
    saved_metadata = json.loads(metadata_bytes.decode("utf-8"))

    if not isinstance(saved_metadata, dict):
        raise ValueError("Saved model metadata must be a JSON object.")

    model_file = saved_metadata.get("model_file")
    saved_feature_columns = saved_metadata.get("feature_columns")

    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("Invalid model file path in metadata.")

    if (
        not isinstance(saved_feature_columns, list)
        or not saved_feature_columns
        or any(
            not isinstance(column, str) or not column.strip()
            for column in saved_feature_columns
        )
        or len(saved_feature_columns) != len(set(saved_feature_columns))
    ):
        raise ValueError("Invalid feature-column schema in metadata.")

    if not isinstance(saved_metadata.get("best_model_name"), str):
        raise ValueError("Invalid best model name in metadata.")

    if not isinstance(saved_metadata.get("target_column"), str):
        raise ValueError("Invalid target column in metadata.")

    candidate_model_path = Path(model_file).expanduser()

    if not candidate_model_path.is_absolute():
        candidate_model_path = PROJECT_ROOT / candidate_model_path

    candidate_model_path = candidate_model_path.resolve()

    if not candidate_model_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Model path points outside the project directory.")

    if not candidate_model_path.is_file():
        raise FileNotFoundError("Saved .joblib model file was not found.")

    saved_pipeline = load_prediction_pipeline(candidate_model_path)

    if not callable(getattr(saved_pipeline, "predict", None)):
        raise ValueError("Saved model is not a usable prediction model.")

    schema_probe = prepare_prediction_features(
        pd.DataFrame(columns=saved_feature_columns),
        saved_pipeline,
        id_column="__estate_ai_schema_check_only__",
    )

    if list(schema_probe.columns) != saved_feature_columns:
        raise ValueError("Saved model feature schema is incompatible.")

    model_stat = candidate_model_path.stat()

    saved_model_signature = hashlib.sha256(
        metadata_bytes
        + (
            f"{candidate_model_path}:{model_stat.st_size}:"
            f"{model_stat.st_mtime_ns}"
        ).encode("utf-8")
    ).hexdigest()

    return {
        "metadata": saved_metadata,
        "model_path": candidate_model_path,
        "feature_columns": saved_feature_columns,
        "pipeline": saved_pipeline,
        "signature": saved_model_signature,
    }

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Universal House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

# Check saved model availability before rendering sidebar navigation.
saved_model_details = None
model_validation_error = None

try:
    saved_model_details = get_saved_model_details()
except Exception as error:
    model_validation_error = str(error)

saved_model_available = saved_model_details is not None
st.session_state.setdefault(
    "_active_model_storage_key", ACTIVE_MODEL_STORAGE_KEY
)

# Restore dashboard data independently of the currently uploaded dataset.
training_artifacts_restored = False
training_artifact_restore_error = None
if "training_results" not in st.session_state:
    training_artifacts_restored, training_artifact_restore_error = (
        restore_training_artifacts()
    )

if saved_model_available:
    active_metadata = saved_model_details["metadata"]
    inferred_name, inferred_unit = infer_target_display_config(
        active_metadata.get("target_column", "Target value"),
        active_metadata.get("training_dataset_name", ""),
        active_metadata.get("feature_columns", []),
    )
    st.session_state.setdefault(
        "_target_display_name",
        active_metadata.get("target_display_name") or inferred_name,
    )
    st.session_state.setdefault(
        "_target_unit",
        active_metadata.get("target_unit") or inferred_unit,
    )

# Values used by Sections 9–12 when no dataset is currently uploaded.
TARGET_DISPLAY_NAME = st.session_state.get("_target_display_name", "Target value")
TARGET_UNIT = target_unit_label(st.session_state.get("_target_unit", "Unspecified units"))

# ==================================================
# PROFESSIONAL APPLICATION SHELL
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --app-accent: #159E9A;
        --app-accent-dark: #117D7A;
        --app-navy: #17324F;
        --app-muted: #718096;
        --app-border: var(--border-color, rgba(120, 135, 155, 0.22));
        --app-surface: var(--secondary-background-color, #F5F7FA);
        --app-bg: var(--background-color, #FFFFFF);
        --app-text: var(--text-color, #1F2937);
    }

    .stApp {
        background:
            radial-gradient(circle at 92% 2%, rgba(21, 158, 154, 0.075), transparent 25rem),
            var(--app-bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewBlockContainer"] {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: var(--app-surface);
        border-right: 1px solid var(--app-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.45;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.4rem 0 1.25rem 0;
        border-bottom: 1px solid var(--app-border);
        margin-bottom: 1.2rem;
    }

    .brand-icon {
        display: grid;
        place-items: center;
        width: 2.55rem;
        height: 2.55rem;
        border-radius: 0.85rem;
        color: #FFFFFF;
        font-size: 1.35rem;
        background: linear-gradient(135deg, #17324F, #159E9A);
        box-shadow: 0 7px 18px rgba(21, 90, 105, 0.18);
    }

    .brand-name {
        color: var(--app-text);
        font-size: 1.02rem;
        font-weight: 800;
        letter-spacing: 0.055em;
        line-height: 1.15;
    }

    .brand-caption {
        color: var(--app-muted);
        font-size: 0.69rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    .sidebar-kicker {
        color: var(--app-muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin: 1rem 0 0.55rem 0;
    }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 0.28rem;
    }

    .sidebar-nav a {
        display: block;
        color: var(--app-text) !important;
        text-decoration: none !important;
        padding: 0.55rem 0.68rem;
        border-radius: 0.55rem;
        border: 1px solid transparent;
        font-size: 0.83rem;
        transition: background 120ms ease, border-color 120ms ease;
    }

    .sidebar-nav a:hover {
        background: rgba(21, 158, 154, 0.09);
        border-color: rgba(21, 158, 154, 0.22);
        color: var(--app-accent-dark) !important;
    }

    .sidebar-note {
        margin-top: 1.25rem;
        padding: 0.8rem;
        border: 1px solid var(--app-border);
        border-radius: 0.75rem;
        background: var(--app-bg);
        color: var(--app-muted);
        font-size: 0.75rem;
        line-height: 1.5;
    }

    .hero-panel {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.75rem 1.9rem;
        margin: 0.15rem 0 1.5rem 0;
        border: 1px solid rgba(21, 158, 154, 0.2);
        border-radius: 1.15rem;
        background: linear-gradient(115deg, rgba(23, 50, 79, 0.98), rgba(20, 83, 100, 0.96) 65%, rgba(21, 158, 154, 0.94));
        box-shadow: 0 12px 28px rgba(20, 50, 75, 0.12);
    }

    .hero-panel:after {
        content: "";
        position: absolute;
        width: 18rem;
        height: 18rem;
        right: -5rem;
        top: -10rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 50%;
        box-shadow: 0 0 0 2rem rgba(255,255,255,0.035), 0 0 0 4rem rgba(255,255,255,0.025);
        pointer-events: none;
    }

    .hero-copy, .hero-chip { position: relative; z-index: 1; }

    .hero-eyebrow {
        color: #A8E7E0;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: clamp(1.55rem, 3vw, 2.15rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.14;
        margin: 0;
    }

    .hero-panel h1.hero-title {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.82);
        font-size: 0.94rem;
        margin: 0.6rem 0 0 0;
        max-width: 48rem;
    }

    .hero-chip {
        flex: 0 0 auto;
        padding: 0.75rem 0.95rem;
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 0.8rem;
        background: rgba(255,255,255,0.10);
        color: #FFFFFF;
        text-align: right;
        backdrop-filter: blur(8px);
    }

    .hero-chip-label {
        display: block;
        color: #A8E7E0;
        font-size: 0.63rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .hero-chip-value {
        display: block;
        font-size: 0.86rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h2 {
        padding-top: 0.35rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid var(--app-border);
        margin-top: 1.55rem;
    }

    h3 { margin-top: 1.1rem; }

    [data-testid="stMetric"] {
        padding: 1rem 1.05rem;
        border: 1px solid var(--app-border);
        border-radius: 0.85rem;
        background: var(--app-surface);
        box-shadow: 0 3px 12px rgba(20, 40, 60, 0.035);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    /* Keep Target Variable card dimensions fixed */
    .st-key-target_variable_metric [data-testid="stMetric"] {
        height: 6.835rem;
        min-height: 6.835rem;
        box-sizing: border-box;
    }

    /* Increase only the target variable text */
    .st-key-target_variable_metric [data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    line-height: 2.33 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed rgba(21, 158, 154, 0.58);
        border-radius: 0.9rem;
        background: rgba(21, 158, 154, 0.035);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--app-border);
        border-radius: 0.75rem;
        overflow: hidden;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 0.62rem;
        font-weight: 700;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .stButton > button[kind="primary"] {
        border: 1px solid var(--app-accent-dark);
        background: linear-gradient(110deg, #167F82, #159E9A);
        color: #FFFFFF;
        box-shadow: 0 5px 14px rgba(21, 158, 154, 0.18);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(20, 50, 70, 0.12);
    }

    [data-testid="stAlert"] {
        border-radius: 0.75rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--app-border);
        border-radius: 0.75rem;
    }

    .dataset-status {
        margin-top: 1rem;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(21, 158, 154, 0.28);
        border-radius: 0.75rem;
        background: rgba(21, 158, 154, 0.07);
    }

    .dataset-status-label {
        display: block;
        color: var(--app-accent-dark);
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .dataset-status-value {
        display: block;
        color: var(--app-text);
        font-size: 0.8rem;
        font-weight: 700;
    }

    .app-footer {
        border-top: 1px solid var(--app-border);
        margin-top: 2.5rem;
        padding: 1rem 0 0.2rem 0;
        color: var(--app-muted);
        font-size: 0.74rem;
        text-align: center;
    }

    @media (max-width: 720px) {
        [data-testid="stAppViewBlockContainer"] {
            padding: 1rem 1rem 2rem 1rem;
        }
        .hero-panel {
            align-items: flex-start;
            flex-direction: column;
            padding: 1.35rem;
        }
        .hero-chip { text-align: left; }
    }

    /* Keep Best Model card height unchanged */
    .st-key-best_model_metric [data-testid="stMetric"] {
        min-height: 6.84rem;
        box-sizing: border-box;
    }

    /* Adjust only the Best Model value text */
    .st-key-best_model_metric [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        line-height: 1.656 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="brand-lockup">
      <div class="brand-icon">⌂</div>
      <div>
        <div class="brand-name">ESTATE AI</div>
        <div class="brand-caption">Property intelligence</div>
      </div>
    </div>
    <div class="sidebar-kicker">Workspace navigation</div>
    <nav class="sidebar-nav">
      <a href="#1-upload-dataset">01&nbsp; · &nbsp;Data intake</a>
      <a href="#2-dataset-overview">02&nbsp; · &nbsp;Dataset overview</a>
      <a href="#3-feature-configuration">03&nbsp; · &nbsp;Feature setup</a>
      <a href="#4-automatic-feature-detection">04&nbsp; · &nbsp;Feature detection</a>
      <a href="#5-configuration-summary">05&nbsp; · &nbsp;Configuration</a>
      <a href="#6-data-quality-analysis">06&nbsp; · &nbsp;Data quality</a>
      <a href="#7-data-quality-summary">07&nbsp; · &nbsp;Quality summary</a>
      <a href="#8-machine-learning-model-training">08&nbsp; · &nbsp;Model training</a>
      <a href="#9-model-training-results">09&nbsp; · &nbsp;Training results</a>
      <a href="#10-model-comparison-dashboard">10&nbsp; · &nbsp;Model analytics</a>
      <a href="#11-model-explainability">11&nbsp; · &nbsp;Explainability</a>
      <a href="#section-12-price-prediction">12&nbsp; · &nbsp;Price prediction</a>
    </nav>
    <div class="sidebar-note">
      <strong>Workflow</strong><br>
      Upload a dataset, validate its quality, train regression models, inspect performance, and generate predictions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-panel">
      <div class="hero-copy">
        <div class="hero-eyebrow">Property intelligence · Machine learning</div>
        <h1 class="hero-title">Universal House Price<br>Prediction System</h1>
        <p class="hero-subtitle">A complete workspace to explore property data, build regression models, understand model behaviour, and estimate house prices.</p>
      </div>
      <div class="hero-chip">
        <span class="hero-chip-label">Platform</span>
        <span class="hero-chip-value">ML Analytics Workspace</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 2. DATASET UPLOAD
# ==================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

dataset_ready = False

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset to access Sections 2–8. Previously saved training results remain available in Sections 9–11.")
else:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        dataset_signature = hashlib.sha256(uploaded_bytes).hexdigest()

        df, is_ames_dataset = load_uploaded_dataset(uploaded_file)

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")
        if not df.columns.is_unique:
            raise ValueError("Dataset contains duplicate column names.")

        # Keep the last successful training dashboard available even when a
        # different dataset is uploaded. It is labelled as a prior run below.
        st.session_state["_dataset_signature"] = dataset_signature

        st.success("Dataset uploaded successfully!")
        st.sidebar.markdown(
            f"""
            <div class="dataset-status">
              <span class="dataset-status-label">Dataset connected</span>
              <span class="dataset-status-value">{df.shape[0]:,} records · {df.shape[1]:,} columns</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_ames_dataset:
            st.caption(
                "Ames Housing format detected. Known absence categories "
                "such as no garage/no pool are preserved separately from "
                "genuinely missing values."
            )

        dataset_ready = True

    except Exception as error:
        st.error(f"Unable to read dataset: {error}")

if dataset_ready:
    # ==================================================
    # 3. DATASET OVERVIEW
    # ==================================================

    st.header("2. Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")


    # ==================================================
    # 4. FEATURE CONFIGURATION
    # ==================================================

    st.header("3. Feature Configuration")

    target_column = st.selectbox(
        "Select Target Column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )

    id_options = [column for column in df.columns if column != target_column]

    id_columns = st.multiselect(
        "Select ID Columns (if any)",
        options=id_options,
    )

    feature_columns = [
        column
        for column in df.columns
        if column != target_column and column not in id_columns
    ]

    X_preview = df[feature_columns]

    numeric_columns = X_preview.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_preview.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    categorical_override = st.multiselect(
        "Treat selected numerical columns as categorical",
        options=numeric_columns,
    )

    categorical_columns += categorical_override
    numeric_columns = [
        column
        for column in numeric_columns
        if column not in categorical_override
    ]

    # Each dataset + target receives its own model directory. This prevents
    # UCI or any future dataset from overwriting the Ames Housing artifacts.
    desired_model_dir, desired_storage_key = dataset_storage_location(
        uploaded_file.name,
        dataset_signature,
        target_column,
        feature_columns,
    )
    current_storage_key = st.session_state.get(
        "_active_model_storage_key", ACTIVE_MODEL_STORAGE_KEY
    )
    if current_storage_key != desired_storage_key or MODEL_DIR.resolve() != desired_model_dir.resolve():
        clear_training_state()
        MODEL_DIR = desired_model_dir
        TRAINING_ARTIFACTS_PATH = MODEL_DIR / "training_artifacts.joblib"
        st.session_state["_active_model_storage_key"] = desired_storage_key
        training_artifacts_restored, training_artifact_restore_error = (
            restore_training_artifacts()
        )

        saved_model_details = None
        model_validation_error = None
        try:
            saved_model_details = get_saved_model_details()
        except Exception as error:
            model_validation_error = str(error)
        saved_model_available = saved_model_details is not None

        # Selecting a dataset with already-saved artifacts makes it the active
        # workspace for the next app restart. Empty, untrained folders are not
        # promoted to active status.
        if training_artifacts_restored or saved_model_available:
            save_active_model_location(
                MODEL_DIR,
                desired_storage_key,
                uploaded_file.name,
            )

    st.sidebar.caption(
        f"Model workspace: {MODEL_DIR.name}"
    )

    inferred_display_name, inferred_target_unit = infer_target_display_config(
        target_column, uploaded_file.name, df.columns.tolist()
    )
    workspace_metadata = {}
    workspace_metadata_path = MODEL_DIR / "best_model_metadata.json"
    workspace_display_config_path = MODEL_DIR / "target_display_config.json"
    if workspace_display_config_path.is_file():
        try:
            workspace_metadata = json.loads(
                workspace_display_config_path.read_text(encoding="utf-8")
            )
        except Exception:
            workspace_metadata = {}
    if workspace_metadata_path.is_file():
        try:
            candidate_metadata = json.loads(
                workspace_metadata_path.read_text(encoding="utf-8")
            )
            if candidate_metadata.get("target_column") == target_column:
                workspace_metadata = {**candidate_metadata, **workspace_metadata}
        except Exception:
            workspace_metadata = {}

    target_name_key = f"target_display_name_{desired_storage_key}_{target_column}"
    target_unit_key = f"target_unit_{desired_storage_key}_{target_column}"
    default_display_name = (
        workspace_metadata.get("target_display_name")
        or st.session_state.get("_target_display_name")
        or inferred_display_name
    )
    default_target_unit = (
        workspace_metadata.get("target_unit")
        or st.session_state.get("_target_unit")
        or inferred_target_unit
    )

    st.subheader("Target Display Settings")
    display_col1, display_col2 = st.columns(2)
    with display_col1:
        target_display_name = st.text_input(
            "Target display name",
            value=default_display_name,
            key=target_name_key,
            help="Friendly name used in charts, explanations and prediction outputs.",
        ).strip() or inferred_display_name
    with display_col2:
        target_unit = st.text_input(
            "Target unit",
            value=default_target_unit,
            key=target_unit_key,
            help="Examples: USD ($), 10,000 TWD / Ping, INR, or tonnes.",
        ).strip() or "Unspecified units"

    st.session_state["_target_display_name"] = target_display_name
    st.session_state["_target_unit"] = target_unit
    TARGET_DISPLAY_NAME = target_display_name
    TARGET_UNIT = target_unit_label(target_unit)
    st.caption(f"Displayed as: {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")

    if st.button(
        "Save target display settings",
        key=f"save_target_display_{desired_storage_key}_{target_column}",
    ):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        display_config = {
            "target_column": target_column,
            "target_display_name": TARGET_DISPLAY_NAME,
            "target_unit": TARGET_UNIT,
            "training_dataset_name": uploaded_file.name,
        }
        temporary_config_path = workspace_display_config_path.with_suffix(".json.tmp")
        temporary_config_path.write_text(
            json.dumps(display_config, indent=4), encoding="utf-8"
        )
        os.replace(temporary_config_path, workspace_display_config_path)
        if workspace_metadata_path.is_file():
            try:
                existing_metadata = json.loads(
                    workspace_metadata_path.read_text(encoding="utf-8")
                )
                if existing_metadata.get("target_column") == target_column:
                    existing_metadata.update({
                        "target_display_name": TARGET_DISPLAY_NAME,
                        "target_unit": TARGET_UNIT,
                    })
                    temporary_metadata_path = workspace_metadata_path.with_suffix(".json.tmp")
                    temporary_metadata_path.write_text(
                        json.dumps(existing_metadata, indent=4), encoding="utf-8"
                    )
                    os.replace(temporary_metadata_path, workspace_metadata_path)
            except Exception as error:
                st.warning(f"Display configuration saved, but model metadata could not be updated: {error}")
        if "training_results" in st.session_state:
            save_training_artifacts()
        st.success("Target display settings saved with this dataset's model workspace.")
        st.rerun()

    # ==================================================
    # 5. AUTOMATIC FEATURE DETECTION AND CONFIGURATION
    # ==================================================

    st.header("4. Automatic Feature Detection")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Numerical Features")
        st.metric("Detected Features", len(numeric_columns))

        with st.expander(
            f"View all {len(numeric_columns)} numerical features"
        ):
            st.dataframe(
                pd.DataFrame({"Feature": numeric_columns}),
                width="stretch",
                hide_index=True,
            )

    with col2:
        st.subheader("Categorical Features")
        st.metric("Detected Features", len(categorical_columns))

        with st.expander(
            f"View all {len(categorical_columns)} categorical features"
        ):
            st.dataframe(
                pd.DataFrame({"Feature": categorical_columns}),
                width="stretch",
                hide_index=True,
            )

    st.header("5. Configuration Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        with st.container(key="target_variable_metric"):
            st.metric("Target Variable", target_column)

    summary_col2.metric("Input Features", len(feature_columns))
    summary_col3.metric("Excluded ID Columns", len(id_columns))

    st.caption(f"Target display: {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")

    summary_col4, summary_col5 = st.columns(2)

    summary_col4.metric("Numerical Features", len(numeric_columns))
    summary_col5.metric("Categorical Features", len(categorical_columns))

    st.caption(
        f"Excluded identifiers: {', '.join(id_columns) if id_columns else 'None'}"
    )

    st.caption(
        f"Categorical overrides: "
        f"{', '.join(categorical_override) if categorical_override else 'None'}"
    )

    st.success("Dataset configuration completed!")

    # ==================================================
    # 6. DATA QUALITY ANALYSIS
    # ==================================================

    st.header("6. Data Quality Analysis")

    missing_count = df.isnull().sum()

    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": missing_count.values,
        "Missing Percentage": (
            missing_count.values / len(df) * 100
        ).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })

    missing_report = missing_report.sort_values(
        "Missing Values",
        ascending=False,
    )

    duplicate_count = int(df.duplicated().sum())

    target_numeric = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    target_missing = int(df[target_column].isna().sum())
    target_invalid = int(
        (df[target_column].notna() & target_numeric.isna()).sum()
    )

    target_numeric_array = target_numeric.to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    target_infinite = int(np.isinf(target_numeric_array).sum())
    target_valid_count = int(np.isfinite(target_numeric_array).sum())
    target_is_valid = (
        target_missing == 0
        and target_invalid == 0
        and target_infinite == 0
        and target_valid_count == len(df)
    )

    quality_col1, quality_col2, quality_col3 = st.columns(3)
    quality_col1.metric("Duplicate Rows", duplicate_count)
    quality_col2.metric(
        "Columns With Missing Values",
        int((missing_count > 0).sum()),
    )
    quality_col3.metric("Valid Numeric Targets", target_valid_count)

    st.subheader("6.1 Missing Value Analysis")
    st.dataframe(
        missing_report,
        width="stretch",
        hide_index=True,
    )

    missing_columns = missing_report[
        missing_report["Missing Values"] > 0
    ]

    if missing_columns.empty:
        st.success("No missing values detected.")
    else:
        st.warning(
            f"{len(missing_columns)} columns contain missing values."
        )

        plot_data = missing_columns.sort_values(
            "Missing Values",
            ascending=True
        )

        fig, ax = plt.subplots(
            figsize=(10, max(3, len(plot_data) * 0.65))
        )

        bars = ax.barh(
            plot_data["Column"],
            plot_data["Missing Values"],
            color="#0F8F8C",
            height=0.55
        )

        ax.bar_label(bars, padding=5, fontsize=10)
        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Features")
        ax.set_title("Missing Value Distribution")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.subheader("6.2 Duplicate Record Analysis")
    if duplicate_count == 0:
        st.success("No duplicate rows detected.")
    else:
        st.warning(f"{duplicate_count} duplicate rows detected.")

    st.subheader("6.3 Data Type Analysis")
    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": df.nunique(dropna=True).values,
    })
    st.dataframe(
        dtype_report,
        width="stretch",
        hide_index=True,
    )

    st.subheader("6.4 Target Column Validation")
    st.write(f"Selected target: **{target_column}** · Display: **{TARGET_DISPLAY_NAME} ({TARGET_UNIT})**")

    target_col1, target_col2, target_col3, target_col4 = st.columns(4)
    target_col1.metric("Missing Targets", target_missing)
    target_col2.metric("Non-numeric Targets", target_invalid)
    target_col3.metric("Infinite Targets", target_infinite)
    target_col4.metric("Valid Targets", target_valid_count)

    if target_is_valid:
        st.success("Target column is valid for regression.")
    else:
        st.error(
            "Target column contains missing, non-numeric or infinite values. "
            "Resolve these issues before training regression models."
        )

    if target_valid_count > 0:
        st.subheader("6.5 Target Statistics")

        finite_target = target_numeric[
            np.isfinite(target_numeric_array)
        ]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        stat_col1.metric(
            f"Average ({TARGET_UNIT})",
            f"{finite_target.mean():,.2f}"
        )

        stat_col2.metric(
            f"Median ({TARGET_UNIT})",
            f"{finite_target.median():,.2f}"
        )

        stat_col3.metric(
            f"Minimum ({TARGET_UNIT})",
            f"{finite_target.min():,.2f}"
        )

        stat_col4.metric(
            f"Maximum ({TARGET_UNIT})",
            f"{finite_target.max():,.2f}"
        )

        fig, ax = plt.subplots(figsize=(11, 4))

        ax.hist(
            finite_target,
            bins=35,
            color="#0F8F8C",
            edgecolor="white",
            alpha=0.9
        )

        ax.axvline(
            finite_target.mean(),
            color="#E58A35",
            linestyle="--",
            linewidth=2,
            label="Mean"
        )

        ax.axvline(
            finite_target.median(),
            color="#264C73",
            linestyle=":",
            linewidth=2,
            label="Median"
        )

        ax.set_title("Target Variable Distribution")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


    # ==================================================
    # 7. FINAL QUALITY SUMMARY
    # ==================================================

    st.header("7. Data Quality Summary")

    total_missing = int(missing_count.sum())
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Total Missing Cells", total_missing)
    summary_col2.metric("Duplicate Rows", duplicate_count)
    summary_col3.metric("Input Features", len(feature_columns))

    st.info(
        "Data quality inspection completed. Review missing values and "
        "target validity before training models."
    )


    # ==================================================
    # 8. MACHINE LEARNING MODEL TRAINING
    # ==================================================

    st.header("8. Machine Learning Model Training")
    st.write(
        "Train and evaluate multiple regression algorithms "
        "using the reusable generic pipeline."
    )

    test_size = st.slider(
        "Validation dataset percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of the dataset reserved for validation instead of model training."
    )

    cv_folds = st.selectbox(
        "Cross-validation folds",
        options=[3, 5],
        index=1,
        help="Number of folds used to evaluate model performance through cross-validation."
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        value=42,
        step=1,
        help="Controls randomness to make model training and data splitting reproducible."
    )

    st.subheader("Select Regression Models")

    model_options = {
        "Linear Regression": "linear_regression",
        "Decision Tree Regressor": "decision_tree",
        "Random Forest Regressor": "random_forest",
    }

    selected_model_labels = st.multiselect(
        "Choose regression models to train",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        help="Select one or more regression algorithms for training."
    )

    selected_models = [
        model_options[label]
        for label in selected_model_labels
    ]


    # Invalidate previous results whenever any setting that affects training changes.
    current_training_signature = (
        dataset_signature,
        target_column,
        tuple(id_columns),
        tuple(categorical_override),
        int(test_size),
        int(cv_folds),
        int(random_state),
        tuple(selected_models),
    )

    if (
        "training_results" in st.session_state
        and st.session_state.get("_training_config_signature") is not None
        and st.session_state.get("_training_config_signature")
        != current_training_signature
    ):
        st.info(
            "The dashboard below shows the last successful saved training run. "
            "Your current dataset or training settings differ; train again only "
            "when you want to replace that saved run."
        )

    validation_rows = int(np.ceil(len(df) * test_size / 100))
    training_rows = len(df) - validation_rows
    training_configuration_valid = (
        target_is_valid
        and len(feature_columns) > 0
        and training_rows >= cv_folds
        and validation_rows >= 2
    )

    if not target_is_valid:
        st.warning("Training is disabled until the target column is valid.")
    elif len(feature_columns) == 0:
        st.warning("Select at least one input feature (not the target or an ID).")
    elif training_rows < cv_folds or validation_rows < 2:
        st.warning(
            f"The dataset is too small for this split and {cv_folds}-fold "
            "cross-validation. Add more records or change the settings."
        )

    train_clicked = st.button(
        "Train Selected Models",
        type="primary",
        disabled=(not selected_models) or (not training_configuration_valid),
    )

    if train_clicked:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Clear old results first so a failed retraining run cannot leave
        # an older model displayed as though it were the current run.
        clear_training_state()

        try:
            X, y = prepare_features_target(
                df,
                target_column=target_column,
                id_columns=id_columns,
                categorical_columns=categorical_override,
            )

            # Treat positive/negative infinity in numeric input features as
            # missing values; the generic pipeline's imputers can then handle them.
            numeric_input_columns = X.select_dtypes(
                include=["number"]
            ).columns
            if len(numeric_input_columns) > 0:
                X.loc[:, numeric_input_columns] = X.loc[
                    :, numeric_input_columns
                ].replace([np.inf, -np.inf], np.nan)

            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=int(random_state),
            )

            if len(X_train) < cv_folds:
                raise ValueError(
                    f"Training set has {len(X_train)} rows, fewer than "
                    f"the {cv_folds} cross-validation folds."
                )

            st.info(
                f"Training records: {len(X_train)} | "
                f"Validation records: {len(X_valid)}"
            )

            results = []
            trained_models = {}
            progress = st.progress(0.0)

            for index, model_name in enumerate(selected_models):
                st.write(f"Training: **{model_name}**")

                pipeline = build_generic_pipeline(
                    X_train,
                    model_name=model_name,
                    categorical_columns=categorical_override,
                    random_state=int(random_state),
                )

                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=1,
                    error_score="raise",
                )

                pipeline.fit(X_train, y_train)

                train_predictions = np.asarray(
                    pipeline.predict(X_train), dtype=float
                ).reshape(-1)
                valid_predictions = np.asarray(
                    pipeline.predict(X_valid), dtype=float
                ).reshape(-1)

                if not np.isfinite(train_predictions).all():
                    raise ValueError(
                        f"{model_name} produced non-finite training predictions."
                    )
                if not np.isfinite(valid_predictions).all():
                    raise ValueError(
                        f"{model_name} produced non-finite validation predictions."
                    )

                train_rmse = float(np.sqrt(mean_squared_error(
                    y_train, train_predictions
                )))
                validation_mse = float(mean_squared_error(
                    y_valid, valid_predictions
                ))
                validation_rmse = float(np.sqrt(validation_mse))
                validation_mae = float(mean_absolute_error(
                    y_valid, valid_predictions
                ))
                validation_r2 = float(r2_score(
                    y_valid, valid_predictions
                ))

                results.append({
                    "Model": model_name,
                    "CV_RMSE_Mean": float(-cv_scores.mean()),
                    "CV_RMSE_Std": float(cv_scores.std()),
                    "Train_RMSE": train_rmse,
                    "Validation_MSE": validation_mse,
                    "Validation_RMSE": validation_rmse,
                    "Validation_MAE": validation_mae,
                    "Validation_R2": validation_r2,
                })

                model_path = MODEL_DIR / f"{model_name}_streamlit.joblib"
                joblib.dump(pipeline, model_path)

                trained_models[model_name] = {
                    "pipeline": pipeline,
                    "predictions": valid_predictions,
                    "model_path": str(model_path),
                }

                progress.progress((index + 1) / len(selected_models))

            # Select and save the best model using validation RMSE.
            results_df = pd.DataFrame(results)
            best_row_index = results_df["Validation_RMSE"].idxmin()
            best_model_name = str(results_df.loc[best_row_index, "Model"])
            best_pipeline = trained_models[best_model_name]["pipeline"]

            best_model_path = MODEL_DIR / "best_house_price_model.joblib"
            joblib.dump(best_pipeline, best_model_path)

            # Save model metadata for persistence after app restart
            metadata = {
                "best_model_name": best_model_name,
                "model_file": str(best_model_path.relative_to(PROJECT_ROOT)),
                "feature_columns": list(X_valid.columns),
                "target_column": target_column,
                "target_display_name": TARGET_DISPLAY_NAME,
                "target_unit": TARGET_UNIT,
                "categorical_columns": list(categorical_override),
                "validation_percentage": float(test_size),
                "cv_folds": int(cv_folds),
                "random_state": int(random_state),
                "selected_models": list(selected_models),
                "training_dataset_name": uploaded_file.name,
                "dataset_storage_key": st.session_state.get(
                    "_active_model_storage_key", "legacy_root"
                ),
                "dataset_signature": dataset_signature,
            }

            metadata_path = MODEL_DIR / "best_model_metadata.json"

            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            st.session_state["training_results"] = results_df
            st.session_state["trained_models"] = trained_models
            st.session_state["validation_actual"] = np.asarray(
                y_valid, dtype=float
            )
            # Save the feature names for future predictions
            st.session_state["feature_columns"] = list(X_valid.columns)
            st.session_state["validation_features"] = X_valid
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_model_path"] = str(best_model_path)
            st.session_state["_training_config_signature"] = (
                current_training_signature
            )
            st.session_state["_training_dataset_signature"] = dataset_signature
            st.session_state["_training_dataset_name"] = uploaded_file.name
            st.session_state["_target_display_name"] = TARGET_DISPLAY_NAME
            st.session_state["_target_unit"] = TARGET_UNIT

            # Persist every model, validation prediction, metric and feature row.
            save_training_artifacts()
            save_active_model_location(
                MODEL_DIR,
                st.session_state.get("_active_model_storage_key", "legacy_root"),
                uploaded_file.name,
            )
            # Refresh the prediction panel's model handle in this same run.
            saved_model_details = get_saved_model_details()
            saved_model_available = True
            model_validation_error = None

            st.success(
                f"Best model saved successfully: {best_model_name}"
            )
            st.info(f"Model location: {best_model_path}")
            st.success("All selected models trained successfully!")

        except Exception as error:
            st.error(f"Model training failed: {error}")

    # One-time migration path for model files saved before training-artifact
    # persistence was introduced. It recreates the dashboard using the
    # existing fitted pipelines and the original training dataset.
    if (
        saved_model_available
        and "training_results" not in st.session_state
        and uploaded_file is not None
    ):
        with st.expander(
            "Recover dashboard from existing trained model files",
            expanded=True,
        ):
            st.write(
                "Your saved model and metadata are available, but the earlier "
                "application did not persist the validation snapshot needed by "
                "Sections 9–11. Upload the original training dataset above and "
                "use the same 20% validation split and random state 42 used in "
                "your previous run. This recovery evaluates the already-fitted "
                "model files; it does not retrain them."
            )
            if st.button(
                "Restore dashboard from existing models",
                key="restore_legacy_dashboard",
                type="primary",
            ):
                try:
                    with st.spinner("Rebuilding the dashboard from saved models..."):
                        recovered_best, recovered_results = rebuild_dashboard_from_existing_models(
                            dataframe=df,
                            target_column=target_column,
                            test_size=test_size,
                            random_state=int(random_state),
                            saved_details=saved_model_details,
                        )
                    save_active_model_location(
                        MODEL_DIR,
                        st.session_state.get("_active_model_storage_key", "legacy_root"),
                        uploaded_file.name,
                    )
                    st.success(
                        f"Dashboard restored from existing fitted models. "
                        f"Best model: {recovered_best}. No model retraining was performed."
                    )
                    st.info(
                        "The original cross-validation scores were not stored in the old files, "
                        "so CV columns are left unavailable in this recovered dashboard."
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Dashboard recovery failed: {error}")


# ==================================================
# 9. TRAINING RESULTS
# ==================================================

if "training_results" in st.session_state:
    st.header("9. Model Training Results")

    results_df = st.session_state["training_results"]

    results_display = results_df.rename(columns={
        "CV_RMSE_Mean": target_metric_label("CV RMSE Mean", TARGET_UNIT),
        "CV_RMSE_Std": target_metric_label("CV RMSE Std", TARGET_UNIT),
        "Train_RMSE": target_metric_label("Training RMSE", TARGET_UNIT),
        "Validation_MSE": target_metric_label("Validation MSE", TARGET_UNIT, squared=True),
        "Validation_RMSE": target_metric_label("Validation RMSE", TARGET_UNIT),
        "Validation_MAE": target_metric_label("Validation MAE", TARGET_UNIT),
        "Validation_R2": "Validation R² (unitless)",
    })
    st.dataframe(
        results_display.round(4),
        width="stretch",
        hide_index=True,
    )

    best_model_name = st.session_state["best_model_name"]
    best_model = results_df.loc[
        results_df["Model"] == best_model_name
    ].iloc[0]

    st.subheader("Best Performing Model")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(key="best_model_metric"):
            st.metric("Best Model", best_model["Model"])

    col2.metric(
        target_metric_label("Validation RMSE", TARGET_UNIT),
        f"{best_model['Validation_RMSE']:,.2f}",
    )
    col3.metric(
        "Validation R²",
        f"{best_model['Validation_R2']:.4f}",
    )

    st.success(f"Best model: {best_model['Model']}")


# ==================================================
# 10. MODEL COMPARISON DASHBOARD
# ==================================================

if "training_results" in st.session_state:
    st.header("10. Model Comparison Dashboard")

    results_df = st.session_state["training_results"]

    # ----------------------------------------------
    # 10.1 PERFORMANCE COMPARISON
    # ----------------------------------------------

    st.subheader("10.1 Model Performance Comparison")

    metric_labels = {
        "Validation_RMSE": target_metric_label("Validation RMSE", TARGET_UNIT),
        "Validation_MAE": target_metric_label("Validation MAE", TARGET_UNIT),
        "Validation_MSE": target_metric_label("Validation MSE", TARGET_UNIT, squared=True),
        "Validation_R2": "Validation R² (unitless)",
        "CV_RMSE_Mean": target_metric_label("Cross-validation RMSE", TARGET_UNIT),
    }
    available_metric_keys = [
        key for key in metric_labels
        if key in results_df.columns
        and pd.to_numeric(results_df[key], errors="coerce").notna().any()
    ]
    metric_choice = st.selectbox(
        "Select evaluation metric",
        options=available_metric_keys,
        format_func=lambda value: metric_labels[value],
        key="comparison_metric",
    )

    model_display_names = {
        "linear_regression": "Linear Regression",
        "decision_tree": "Decision Tree Regressor",
        "random_forest": "Random Forest Regressor",
    }
    comparison_plot = results_df[["Model", metric_choice]].copy()
    comparison_plot["Model_Display"] = comparison_plot["Model"].map(
        lambda value: model_display_names.get(value, value.replace("_", " ").title())
    )
    comparison_plot = comparison_plot.sort_values(metric_choice, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(
        comparison_plot["Model_Display"],
        comparison_plot[metric_choice],
        color="#0F8F8C",
        height=0.58,
    )
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9)
    ax.set_title(f"{metric_labels[metric_choice]} by Regression Model", pad=14)
    ax.set_xlabel(metric_labels[metric_choice])
    ax.set_ylabel("Regression Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if metric_choice == "Validation_R2":
        st.caption("Higher R² is better. For error metrics (RMSE, MAE and MSE), lower values are better.")
    else:
        st.caption("Lower values indicate smaller prediction errors. R² is interpreted differently: higher is better.")

    # ----------------------------------------------
    # 10.2 COMPLETE MODEL COMPARISON
    # ----------------------------------------------

    st.subheader("10.2 Complete Model Comparison")

    comparison_columns = [
        "Model",
        "CV_RMSE_Mean",
        "Train_RMSE",
        "Validation_RMSE",
        "Validation_MAE",
        "Validation_R2",
    ]
    comparison_display = results_df[comparison_columns].copy()
    comparison_display["Model"] = comparison_display["Model"].map(
        lambda value: model_display_names.get(value, value.replace("_", " ").title())
    )
    comparison_display = comparison_display.rename(columns={
        "CV_RMSE_Mean": target_metric_label("CV RMSE (Mean)", TARGET_UNIT),
        "Train_RMSE": target_metric_label("Training RMSE", TARGET_UNIT),
        "Validation_RMSE": target_metric_label("Validation RMSE", TARGET_UNIT),
        "Validation_MAE": target_metric_label("Validation MAE", TARGET_UNIT),
        "Validation_R2": "Validation R² (unitless)",
    })
    st.dataframe(
        comparison_display.round(3),
        width="stretch",
        hide_index=True,
    )

    # ----------------------------------------------
    # 10.3 ACTUAL VS PREDICTED
    # ----------------------------------------------

    st.subheader("10.3 Actual vs Predicted Prices")

    trained_models = st.session_state["trained_models"]
    model_names = list(trained_models.keys())

    selected_model = st.selectbox(
        "Choose a model",
        options=model_names,
        key="actual_predicted_model",
    )

    actual_values = np.asarray(
        st.session_state["validation_actual"], dtype=float
    ).reshape(-1)
    predicted_values = np.asarray(
        trained_models[selected_model]["predictions"], dtype=float
    ).reshape(-1)

    arrays_are_valid = (
        len(actual_values) == len(predicted_values)
        and len(actual_values) > 0
        and np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all()
    )

    if not arrays_are_valid:
        st.error(
            "Actual and predicted values are not aligned or contain "
            "invalid numbers. Retrain the models before using this dashboard."
        )
    else:
        prediction_df = pd.DataFrame({
            "Actual Price": actual_values,
            "Predicted Price": predicted_values,
        })

        fig, ax = plt.subplots(figsize=(9.5, 6))
        ax.scatter(
            actual_values,
            predicted_values,
            color="#0F8F8C",
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
            s=42,
        )
        reference_min = float(min(actual_values.min(), predicted_values.min()))
        reference_max = float(max(actual_values.max(), predicted_values.max()))
        ax.plot(
            [reference_min, reference_max],
            [reference_min, reference_max],
            color="#D97757",
            linestyle="--",
            linewidth=2,
            label="Perfect prediction",
        )
        ax.set_title(
            f"Actual vs Predicted {TARGET_DISPLAY_NAME} — "
            f"{model_display_names.get(selected_model, selected_model)}",
            pad=14,
        )
        ax.set_xlabel(f"Actual {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
        ax.set_ylabel(f"Predicted {TARGET_DISPLAY_NAME} ({TARGET_UNIT})")
        ax.ticklabel_format(style="plain", axis="both")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        display_predictions = prediction_df.head(20).copy()
        display_predictions = display_predictions.rename(columns={
            "Actual Price": f"Actual {TARGET_DISPLAY_NAME} ({TARGET_UNIT})",
            "Predicted Price": f"Predicted {TARGET_DISPLAY_NAME} ({TARGET_UNIT})",
        })
        st.dataframe(
            display_predictions.round(2),
            width="stretch",
            hide_index=True,
        )

        if np.any(predicted_values < 0):
            if selected_model == "linear_regression":
                st.warning(
                    "Linear Regression produced negative house-price "
                    "predictions. This can happen with a linear model; "
                    "inspect the affected records and compare other models."
                )
            else:
                st.warning(
                    "This model produced negative predictions. For a "
                    "tree-based model trained on positive targets, inspect "
                    "the data and prediction pipeline carefully."
                )

        # ----------------------------------------------
        # 10.4 RESIDUAL ANALYSIS
        # ----------------------------------------------

        st.subheader("10.4 Residual Analysis")

        residuals = actual_values - predicted_values

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Mean Residual ({TARGET_UNIT})", f"{np.mean(residuals):,.2f}")
        col2.metric(f"Minimum Residual ({TARGET_UNIT})", f"{np.min(residuals):,.2f}")
        col3.metric(f"Maximum Residual ({TARGET_UNIT})", f"{np.max(residuals):,.2f}")

        st.markdown("#### Residual Distribution")
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.hist(
            residuals,
            bins=30,
            color="#0F8F8C",
            edgecolor="white",
            alpha=0.88,
        )
        ax.axvline(
            0,
            color="#D97757",
            linestyle="--",
            linewidth=2,
            label="Zero error",
        )
        ax.axvline(
            float(np.mean(residuals)),
            color="#17324F",
            linestyle=":",
            linewidth=2,
            label=f"Mean residual: {np.mean(residuals):,.0f}",
        )
        ax.set_xlabel(f"Residual (Actual − Predicted) ({TARGET_UNIT})")
        ax.set_ylabel("Number of Homes")
        ax.set_title(f"Residual Distribution — {model_display_names.get(selected_model, selected_model)}")
        ax.ticklabel_format(style="plain", axis="x")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.write("Residual Outlier Investigation")
        residual_check = pd.DataFrame({
            "Actual": actual_values,
            "Predicted": predicted_values,
            "Residual": residuals,
            "Absolute_Error": np.abs(residuals),
        })

        residual_display = residual_check.sort_values(
            "Absolute_Error", ascending=False
        ).head(10).rename(columns={
            "Actual": f"Actual ({TARGET_UNIT})",
            "Predicted": f"Predicted ({TARGET_UNIT})",
            "Residual": f"Residual ({TARGET_UNIT})",
            "Absolute_Error": f"Absolute Error ({TARGET_UNIT})",
        })
        st.dataframe(
            residual_display.round(2),
            width="stretch",
            hide_index=True,
        )

    # ----------------------------------------------
    # 10.5 BEST MODEL SUMMARY
    # ----------------------------------------------

    st.subheader("10.5 Best Model Summary")

    best_model_name = st.session_state["best_model_name"]
    best_result = results_df.loc[
        results_df["Model"] == best_model_name
    ].iloc[0]

    st.success(f"Best performing model: {model_display_names.get(best_result['Model'], best_result['Model'])}")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        target_metric_label("Validation RMSE", TARGET_UNIT),
        f"{best_result['Validation_RMSE']:,.2f}",
    )
    col2.metric(
        target_metric_label("Validation MAE", TARGET_UNIT),
        f"{best_result['Validation_MAE']:,.2f}",
    )
    col3.metric(
        "Validation R²",
        f"{best_result['Validation_R2']:.4f}",
    )

    # ----------------------------------------------
    # 10.6 DOWNLOAD EVALUATION RESULTS
    # ----------------------------------------------

    if arrays_are_valid:
        st.subheader("10.6 Download Evaluation Results")

        evaluation_df = residual_check.rename(columns={
            "Actual": "Actual_Price",
            "Predicted": "Predicted_Price",
            "Residual": "Prediction_Error",
            "Absolute_Error": "Absolute_Error",
        })
        evaluation_column_names = {
            "Actual_Price": f"Actual {TARGET_DISPLAY_NAME} ({TARGET_UNIT})",
            "Predicted_Price": f"Predicted {TARGET_DISPLAY_NAME} ({TARGET_UNIT})",
            "Prediction_Error": f"Prediction Error ({TARGET_UNIT})",
            "Absolute_Error": f"Absolute Error ({TARGET_UNIT})",
        }
        evaluation_export = evaluation_df.rename(columns=evaluation_column_names)
        evaluation_display = evaluation_export.head(10)
        st.dataframe(
            evaluation_display.round(2),
            width="stretch",
            hide_index=True,
        )

        csv_data = evaluation_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Evaluation Results",
            data=csv_data,
            file_name=f"house_price_evaluation_{selected_model}.csv",
            mime="text/csv",
        )


# ============================================================
# 11. MODEL EXPLAINABILITY - FEATURE IMPORTANCE
# ============================================================

if "training_results" in st.session_state:
    st.header("11. Model Explainability")

    st.subheader("11.1 Global Feature Importance")

    st.write(
        "Identify which original house features have "
        "the greatest influence on model performance."
    )

    best_model_name = st.session_state["best_model_name"]

    best_pipeline = st.session_state["trained_models"][
        best_model_name
    ]["pipeline"]

    X_valid = st.session_state["validation_features"]
    y_valid = st.session_state["validation_actual"]

    if st.button("Calculate Feature Importance"):

        with st.spinner("Calculating feature importance..."):

            try:
                importance_df = calculate_permutation_importance(
                    pipeline=best_pipeline,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    n_repeats=5,
                    random_state=42
                )

                st.session_state["feature_importance"] = importance_df
                save_training_artifacts()

                st.success("Feature importance calculated successfully!")

            except Exception as error:
                st.error(f"Feature importance calculation failed: {error}")

    if "feature_importance" in st.session_state:

        importance_df = st.session_state["feature_importance"]

        st.subheader("Feature Importance Ranking")

        st.dataframe(
            importance_df.round(3),
            width="stretch",
            hide_index=True
        )

        st.subheader("Top 15 Important Features")

        top_features = importance_df.head(15).sort_values(
            "Importance_Mean",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            top_features["Feature"],
            top_features["Importance_Mean"],
            color="#0F8F8C",
            height=0.65
        )

        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)

        ax.set_title("Top 15 Features by Permutation Importance")
        ax.set_xlabel("Mean Importance")
        ax.set_ylabel("Feature")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


    # --------------------------------------------------------
    # 11.2 SHAP EXPLAINABILITY
    # --------------------------------------------------------

    st.subheader("11.2 SHAP Explainability")

    st.write(
        "Understand how individual features contribute "
        f"to the model's {TARGET_DISPLAY_NAME.lower()} predictions."
    )

    if st.button("Calculate SHAP Explanations"):

        with st.spinner("Calculating SHAP values..."):

            try:
                shap_result = calculate_shap_values(
                    pipeline=best_pipeline,
                    X_background=X_valid,
                    X_explain=X_valid
                )

                # Store a compact, portable representation instead of the
                # full SHAP object so it can be restored after an app restart.
                st.session_state["shap_values"] = {
                    "values": np.asarray(shap_result.values),
                    "feature_names": [str(name) for name in shap_result.feature_names],
                }
                save_training_artifacts()

                st.success("SHAP explanations calculated successfully!")

            except Exception as error:
                st.error(f"SHAP calculation failed: {error}")

    if "shap_values" in st.session_state:

        shap_values = st.session_state["shap_values"]
        shap_array = np.asarray(shap_values["values"])
        shap_feature_names = list(shap_values["feature_names"])

        # Global SHAP importance
        st.markdown("#### Global SHAP Feature Importance")

        shap_importance = pd.DataFrame({
            "Feature": shap_feature_names,
            "Mean_Absolute_SHAP": np.abs(shap_array).mean(axis=0)
        })

        shap_importance = shap_importance.sort_values(
            by="Mean_Absolute_SHAP",
            ascending=False
        ).reset_index(drop=True)

        shap_importance_display = shap_importance.head(15).rename(columns={
            "Mean_Absolute_SHAP": f"Mean Absolute SHAP ({TARGET_UNIT})",
        })
        st.dataframe(
            shap_importance_display.round(3),
            width="stretch",
            hide_index=True
        )

        global_shap_top = shap_importance.head(15).sort_values(
            "Mean_Absolute_SHAP",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            global_shap_top["Feature"],
            global_shap_top["Mean_Absolute_SHAP"],
            color="#0F8F8C",
            height=0.65
        )

        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)

        ax.set_title("Top 15 Features by Global SHAP Importance")
        ax.set_xlabel(f"Mean Absolute SHAP Value ({TARGET_UNIT})")
        ax.set_ylabel("Feature")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # Individual prediction explanation
        st.markdown("#### Individual House Prediction Explanation")

        selected_row = st.selectbox(
            "Select a house record to explain",
            options=list(range(len(shap_array))),
            format_func=lambda x: f"House Record {x + 1}"
        )

        local_explanation = pd.DataFrame({
            "Feature": shap_feature_names,
            "SHAP_Value": shap_array[selected_row]
        })

        local_explanation["Absolute_SHAP"] = (
            local_explanation["SHAP_Value"].abs()
        )

        local_explanation = local_explanation.sort_values(
            by="Absolute_SHAP",
            ascending=False
        ).head(15)

        st.write(
            "Positive SHAP values push the prediction higher; "
            "negative values push it lower."
        )

        local_explanation_display = local_explanation.rename(columns={
            "SHAP_Value": f"SHAP Value ({TARGET_UNIT})",
            "Absolute_SHAP": f"Absolute SHAP ({TARGET_UNIT})",
        })
        st.dataframe(
            local_explanation_display.round(3),
            width="stretch",
            hide_index=True
        )

        local_explanation_chart = local_explanation.sort_values(
            "SHAP_Value",
            ascending=True
        )

        bar_colors = [
            "#0F8F8C" if value >= 0 else "#D97757"
            for value in local_explanation_chart["SHAP_Value"]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            local_explanation_chart["Feature"],
            local_explanation_chart["SHAP_Value"],
            color=bar_colors,
            height=0.65
        )

        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)

        ax.axvline(0, color="#64748B", linewidth=1)
        ax.set_title(f"How Features Influence This {TARGET_DISPLAY_NAME} Prediction")
        ax.set_xlabel(f"SHAP Value ({TARGET_UNIT})")
        ax.set_ylabel("Feature")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# If no dashboard snapshot exists yet, keep real section anchors visible.
if "training_results" not in st.session_state:
    st.header("9. Model Training Results")
    if saved_model_available:
        st.info(
            "A trained model is saved, but the earlier version did not save "
            "the validation results required for this dashboard. Upload the "
            "original training dataset and use the recovery control in Section 8."
        )
        if training_artifact_restore_error:
            st.warning(
                f"The saved dashboard snapshot could not be restored: "
                f"{training_artifact_restore_error}"
            )
    else:
        st.info("Train a regression model in Section 8 to view training results here.")

    st.header("10. Model Comparison Dashboard")
    st.info("Model comparison charts will appear here after training results are available.")

    st.header("11. Model Explainability")
    st.info("Feature importance and SHAP explanations will appear here after a model is trained.")
else:
    saved_run_name = st.session_state.get(
        "_training_dataset_name", "Previously trained dataset"
    )
    st.caption(f"Dashboard data source: {saved_run_name} · Persisted training run")

# --------------------------------------------------------
# 12. HOUSE PRICE PREDICTION INTERFACE
# --------------------------------------------------------

st.header(
    "12. Price prediction",
    anchor="section-12-price-prediction"
)


st.write(
    "Upload a dataset to generate predictions using the best previously "
    "saved regression model. You do not need to retrain when a valid model "
    "is already available."
)

saved_metadata = None
saved_model_path = None
saved_feature_columns = None
saved_pipeline = None
saved_model_signature = None

if saved_model_available:
    saved_metadata = saved_model_details["metadata"]
    saved_model_path = saved_model_details["model_path"]
    saved_feature_columns = saved_model_details["feature_columns"]
    saved_pipeline = saved_model_details["pipeline"]
    saved_model_signature = saved_model_details["signature"]
    saved_target_display_name, inferred_saved_unit = infer_target_display_config(
        saved_metadata.get("target_column", "Target value"),
        saved_metadata.get("training_dataset_name", ""),
        saved_metadata.get("feature_columns", []),
    )
    saved_target_display_name = (
        saved_metadata.get("target_display_name") or saved_target_display_name
    )
    saved_target_unit = target_unit_label(
        saved_metadata.get("target_unit") or inferred_saved_unit
    )
else:
    saved_target_display_name = TARGET_DISPLAY_NAME
    saved_target_unit = TARGET_UNIT

if saved_pipeline is None:
    st.warning(
        "**Prediction Access Restricted**\n\n"
        "To generate predictions, you must first train a "
        "machine learning model.\n\n"
        "Navigate to **Section 8 – Model Training**, complete training, "
        "and ensure that the trained model is saved successfully."
    )
    if model_validation_error:
        st.caption(f"Model availability check: {model_validation_error}")
else:
    st.success(
        f"Saved model ready: **{saved_metadata['best_model_name']}**. "
        "You can generate predictions without retraining."
    )

        # --------------------------------------------------------
    # INDIVIDUAL PROPERTY PREDICTION
    # --------------------------------------------------------

    st.markdown("### Individual Property Prediction")
    st.write(
        "Enter property details to generate a prediction "
        "using the currently saved machine learning model."
    )

    # Identify numerical and categorical features from the fitted model.
    preprocessor = saved_pipeline.named_steps.get("preprocessor")

    individual_submitted = False

    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        st.warning(
            "Unable to inspect the saved model's feature schema. "
            "Bulk Prediction remains available."
        )
    else:
        categorical_features = set()
        category_options = {}

        for name, transformer, columns in preprocessor.transformers_:
            if isinstance(columns, str):
                columns = [columns]
            else:
                columns = list(columns)

            if name == "categorical":
                categorical_features.update(columns)

                encoder = None
                if hasattr(transformer, "named_steps"):
                    encoder = next(
                        (
                            step
                            for step in transformer.named_steps.values()
                            if hasattr(step, "categories_")
                        ),
                        None,
                    )
                elif hasattr(transformer, "categories_"):
                    encoder = transformer

                if encoder is not None:
                    category_options = {
                        **category_options,
                        **{
                            feature: [
                                value.item()
                                if isinstance(value, np.generic)
                                else value
                                for value in categories
                                if pd.notna(value)
                            ]
                            for feature, categories in zip(
                                columns, encoder.categories_
                            )
                        },
                    }

        reference_data = st.session_state.get("validation_features")
        if (
            not isinstance(reference_data, pd.DataFrame)
            or not set(saved_feature_columns).issubset(reference_data.columns)
        ):
            reference_data = None

        uci_labels = {
            "X1": "Transaction date (decimal year)",
            "X2": "House age (years)",
            "X3": "Distance to nearest MRT station (meters)",
            "X4": "Number of convenience stores",
            "X5": "Latitude",
            "X6": "Longitude",
        }

        individual_inputs = {}

        with st.container(border=True):
            with st.form("individual_prediction_form"):

                st.markdown("#### Numerical Features")

                numeric_features = [
                    feature
                    for feature in saved_feature_columns
                    if feature not in categorical_features
                ]

                numeric_columns = st.columns(2)

                for index, feature in enumerate(numeric_features):
                    sample = (
                        pd.to_numeric(
                            reference_data[feature],
                            errors="coerce",
                        ).dropna()
                        if reference_data is not None
                        else pd.Series(dtype=float)
                    )

                    default_value = (
                        float(sample.median()) if not sample.empty else 0.0
                    )

                    integer_like = (
                        not sample.empty
                        and np.isclose(sample % 1, 0).all()
                    )

                    if feature in {"X5", "X6"}:
                        step, number_format = 0.000001, "%.6f"
                    elif feature == "X1":
                        step, number_format = 0.0001, "%.4f"
                    elif integer_like:
                        step, number_format = 1.0, "%.0f"
                    else:
                        step, number_format = 0.01, "%.2f"

                    label = uci_labels.get(
                        feature,
                        feature.replace("_", " ").title(),
                    )

                    with numeric_columns[index % 2]:
                        individual_inputs[feature] = st.number_input(
                            label,
                            value=default_value,
                            step=step,
                            format=number_format,
                            key=(
                                f"individual_num_"
                                f"{saved_model_signature[:10]}_{index}"
                            ),
                        )

                if categorical_features:
                    st.markdown("#### Categorical Features")

                    categorical_columns = st.columns(2)

                    for index, feature in enumerate(
                        f for f in saved_feature_columns
                        if f in categorical_features
                    ):
                        options = category_options.get(feature, [])

                        mode = (
                            reference_data[feature].mode(dropna=True)
                            if reference_data is not None
                            else pd.Series(dtype=object)
                        )

                        default_value = (
                            mode.iloc[0] if not mode.empty else None
                        )

                        with categorical_columns[index % 2]:
                            if options:
                                selected_index = (
                                    options.index(default_value)
                                    if default_value in options else 0
                                )

                                individual_inputs[feature] = st.selectbox(
                                    feature.replace("_", " ").title(),
                                    options,
                                    index=selected_index,
                                    key=(
                                        f"individual_cat_"
                                        f"{saved_model_signature[:10]}_{index}"
                                    ),
                                )
                            else:
                                individual_inputs[feature] = st.text_input(
                                    feature.replace("_", " ").title(),
                                    value=(
                                        str(default_value)
                                        if default_value is not None
                                        else ""
                                    ),
                                    key=(
                                        f"individual_text_"
                                        f"{saved_model_signature[:10]}_{index}"
                                    ),
                                )

                individual_submitted = st.form_submit_button(
                    "Predict Individual Property Price",
                    type="primary",
                    width="stretch",
                )

        if individual_submitted:
            st.session_state.pop("individual_prediction_result", None)

            try:
                individual_input_df = pd.DataFrame(
                    [[individual_inputs[f] for f in saved_feature_columns]],
                    columns=saved_feature_columns,
                )

                individual_id_column = (
                    "__estate_ai_no_identifier__"
                    if "Id" in saved_feature_columns
                    else "Id"
                )

                with st.spinner("Generating individual prediction..."):
                    prepare_prediction_features(
                        individual_input_df,
                        saved_pipeline,
                        id_column=individual_id_column,
                    )

                    individual_prediction = np.asarray(
                        predict_house_prices(
                            pipeline=saved_pipeline,
                            data=individual_input_df,
                            id_column=individual_id_column,
                        ),
                        dtype=float,
                    ).reshape(-1)

                if (
                    individual_prediction.size != 1
                    or not np.isfinite(individual_prediction).all()
                ):
                    raise ValueError(
                        "The model did not return one valid prediction."
                    )

                st.session_state["individual_prediction_result"] = {
                    "signature": saved_model_signature,
                    "inputs": individual_input_df,
                    "value": float(individual_prediction[0]),
                }

            except Exception as error:
                st.error(f"Individual prediction failed: {error}")

        result = st.session_state.get("individual_prediction_result")

        if (
            isinstance(result, dict)
            and result.get("signature") == saved_model_signature
        ):
            st.markdown("#### Individual Prediction Result")

            st.metric(
                f"Predicted {saved_target_display_name} ({saved_target_unit})",
                f"{result['value']:,.2f}",
            )
            result_table = result["inputs"].copy()
            result_table[
                f"Predicted {saved_target_display_name} ({saved_target_unit})"
            ] = result["value"]

            with st.expander("View Property Details"):
                st.dataframe(
                    result_table,
                    width="stretch",
                    hide_index=True,
                )

            st.download_button(
                "Download Individual Prediction",
                data=result_table.to_csv(index=False).encode("utf-8"),
                file_name="individual_house_price_prediction.csv",
                mime="text/csv",
                key="download_individual_prediction",
            )

    st.markdown("### Upload Your House Dataset")

    with st.container(border=True):
        st.write(
            "Upload a CSV or Excel dataset containing the features used "
            "during training. The application validates the feature schema "
            "before generating predictions."
        )

        uploaded_prediction_file = st.file_uploader(
            "Upload Housing Dataset",
            type=["csv", "xlsx", "xls"],
            key="prediction_upload",
            help="Supported formats: CSV, Excel (.xlsx and .xls).",
        )

    if uploaded_prediction_file is not None:
        try:
            prediction_file_bytes = uploaded_prediction_file.getvalue()
            prediction_file_signature = hashlib.sha256(
                prediction_file_bytes
            ).hexdigest()
            prediction_run_signature = (
                f"{prediction_file_signature}:{saved_model_signature}"
            )

            # Clear stale output when either the uploaded file or saved model changes.
            if (
                st.session_state.get("_prediction_results_signature")
                != prediction_run_signature
            ):
                st.session_state.pop("prediction_results", None)
                st.session_state.pop("prediction_values", None)
                st.session_state.pop("prediction_file_name", None)

            file_extension = Path(uploaded_prediction_file.name).suffix.lower()
            if file_extension == ".csv":
                prediction_df = pd.read_csv(io.BytesIO(prediction_file_bytes))
            elif file_extension in [".xlsx", ".xls"]:
                prediction_df = pd.read_excel(io.BytesIO(prediction_file_bytes))
            else:
                raise ValueError("Unsupported file format.")

            if prediction_df.empty:
                raise ValueError("The uploaded prediction dataset is empty.")
            if not prediction_df.columns.is_unique:
                raise ValueError("Dataset contains duplicate column names.")

            st.markdown("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", f"{len(prediction_df):,}")
            col2.metric("Available Columns", f"{len(prediction_df.columns):,}")
            col3.metric("Missing Values", f"{prediction_df.isna().sum().sum():,}")

            with st.expander("Preview Uploaded Dataset"):
                st.dataframe(
                    prediction_df.head(10),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown("### Generate Predictions")
            if st.button(
                f"Generate {saved_target_display_name} Predictions",
                type="primary",
                width="stretch",
            ):
                missing_features = [
                    feature for feature in saved_feature_columns
                    if feature not in prediction_df.columns
                ]
                if missing_features:
                    st.error(
                        f"Dataset is missing {len(missing_features)} features "
                        "required by the trained model."
                    )
                    with st.expander("View Missing Features"):
                        st.write(missing_features)
                else:
                    try:
                        with st.spinner("Generating house-price predictions..."):
                            # If Id was a model input, preserve it; otherwise
                            # allow an optional Id identifier to be ignored.
                            prediction_id_column = (
                                "__estate_ai_no_identifier__"
                                if "Id" in saved_feature_columns
                                else "Id"
                            )
                            prepare_prediction_features(
                                prediction_df,
                                saved_pipeline,
                                id_column=prediction_id_column,
                            )
                            predictions = predict_house_prices(
                                pipeline=saved_pipeline,
                                data=prediction_df,
                                id_column=prediction_id_column,
                            )

                            prediction_results = prediction_df.copy().reset_index(drop=True)
                            prediction_column_name = (
                                f"Predicted {saved_target_display_name} ({saved_target_unit})"
                            )
                            prediction_results[prediction_column_name] = predictions

                        st.session_state["prediction_results"] = prediction_results
                        st.session_state["prediction_values"] = np.asarray(
                            predictions, dtype=float
                        )
                        st.session_state["prediction_file_name"] = uploaded_prediction_file.name
                        st.session_state["_prediction_results_signature"] = prediction_run_signature
                        st.success("House-price predictions generated successfully!")

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

            # Re-render saved results after Streamlit reruns while the same
            # uploaded file and same saved model remain selected.
            results_are_current = (
                "prediction_results" in st.session_state
                and st.session_state.get("_prediction_results_signature")
                == prediction_run_signature
            )

            if results_are_current:
                prediction_results = st.session_state["prediction_results"]
                predictions = np.asarray(
                    st.session_state["prediction_values"], dtype=float
                )

                if len(predictions) > 0 and np.isfinite(predictions).all():
                    st.markdown("### Prediction Summary")
                                        # First row: Total predictions and average
                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Total Predictions",
                        f"{len(predictions):,}",
                    )
                    col2.metric(
                        f"Average Predicted {saved_target_display_name} ({saved_target_unit})",
                        f"{np.mean(predictions):,.2f}",
                    )

                    # Second row: Minimum and maximum
                    col3, col4 = st.columns(2)

                    col3.metric(
                        f"Minimum Predicted {saved_target_display_name} ({saved_target_unit})",
                        f"{np.min(predictions):,.2f}",
                    )
                    col4.metric(
                        f"Maximum Predicted {saved_target_display_name} ({saved_target_unit})",
                        f"{np.max(predictions):,.2f}",
                    )

                    st.markdown("### Predicted Price Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4.5))
                    ax.hist(
                        predictions,
                        bins=20,
                        color="#0F8F8C",
                        edgecolor="white",
                        alpha=0.9,
                    )
                    ax.axvline(
                        np.mean(predictions),
                        color="#D97757",
                        linestyle="--",
                        linewidth=2,
                        label=f"Average: {np.mean(predictions):,.2f} {saved_target_unit}",
                    )
                    ax.axvline(
                        np.median(predictions),
                        color="#112B46",
                        linestyle=":",
                        linewidth=2,
                        label=f"Median: {np.median(predictions):,.2f} {saved_target_unit}",
                    )
                    ax.set_xlabel(f"Predicted {saved_target_display_name} ({saved_target_unit})")
                    ax.set_ylabel("Number of Records")
                    ax.set_title(f"Distribution of Predicted {saved_target_display_name}")
                    ax.legend()
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(axis="y", alpha=0.2)
                    ax.set_axisbelow(True)
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                    st.markdown("### Prediction Results")
                    st.dataframe(
                        prediction_results,
                        width="stretch",
                        hide_index=True,
                    )
                    csv_output = prediction_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_output,
                        file_name="house_price_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        width="stretch",
                        key="download_prediction_results",
                    )
                else:
                    st.error("Stored prediction results are empty or invalid. Generate predictions again.")

        except Exception as error:
            st.error(f"Unable to read prediction dataset: {error}")


# --------------------------------------------------------
# APPLICATION FOOTER
# --------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
      ESTATE AI &nbsp;·&nbsp; Universal House Price Prediction System
      <br>Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)