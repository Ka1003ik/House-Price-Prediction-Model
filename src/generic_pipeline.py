
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_target(
    df,
    target_column,
    id_columns=None,
    categorical_columns=None
):
    """
    Prepare a dataset using configurable target and ID columns.

    Automatically retains numerical and categorical features.
    """

    id_columns = list(id_columns or [])
    categorical_columns = list(categorical_columns or [])

    if target_column not in df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    missing_ids = [
        col for col in id_columns
        if col not in df.columns
    ]

    if missing_ids:
        raise ValueError(
            f"Identifier columns not found: {missing_ids}"
        )

    missing_categories = [
        col for col in categorical_columns
        if col not in df.columns
    ]

    if missing_categories:
        raise ValueError(
            f"Categorical columns not found: {missing_categories}"
        )

    # Convert target to numeric.
    y = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    # Do not silently discard records with missing targets.
    if y.isna().any():
        raise ValueError(
            f"Target contains {y.isna().sum()} missing or invalid values."
        )

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError(
            "Target contains infinite values."
        )

    columns_to_remove = [
        target_column,
        *id_columns
    ]

    X = df.drop(
        columns=columns_to_remove
    ).copy()

    if X.empty:
        raise ValueError("No input features remain after exclusions.")

    # Explicitly configured categorical features are converted
    # to consistent text values, including integer-coded categories.
    for column in categorical_columns:
        X[column] = X[column].astype("object")

    return X, y


# ============================================================
# 2. DETECT FEATURE TYPES
# ============================================================

def detect_feature_types(
    X,
    categorical_columns=None
):
    """
    Automatically identify numerical and categorical features.

    Additional columns can be forced to categorical when
    numeric codes represent labels rather than quantities.
    """

    categorical_columns = set(categorical_columns or [])

    missing_columns = categorical_columns.difference(X.columns)

    if missing_columns:
        raise ValueError(
            f"Unknown categorical columns: {sorted(missing_columns)}"
        )

    detected_categorical = (
        X.select_dtypes(
            include=["object", "category", "bool", "string"]
        )
        .columns
        .tolist()
    )

    categorical_features = list(
        dict.fromkeys(
            detected_categorical +
            list(categorical_columns)
        )
    )

    numeric_features = [
        column
        for column in X.select_dtypes(include=["number"]).columns
        if column not in categorical_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable numerical or categorical features.")

    return numeric_features, categorical_features


# ============================================================
# 3. BUILD GENERIC PREPROCESSOR
# ============================================================

def build_generic_preprocessor(
    X_train,
    categorical_columns=None
):
    """
    Build a reusable preprocessing transformer.

    Numerical:
        Median imputation + standard scaling.

    Categorical:
        Unknown-value imputation + one-hot encoding.
    """

    numeric_features, categorical_features = (
        detect_feature_types(
            X_train,
            categorical_columns=categorical_columns
        )
    )

    transformers = []

    # Numerical pipeline.
    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numeric_pipeline,
                numeric_features
            )
        )

    # Categorical pipeline.
    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# 4. CREATE REGRESSION MODEL
# ============================================================

def create_regression_model(
    model_name,
    random_state=42,
    model_params=None
):
    """
    Create a regression algorithm by name.

    Supported:
        linear_regression
        decision_tree
        random_forest

    Optional model_params can override default settings.
    """

    model_params = dict(model_params or {})

    models = {
        "linear_regression": LinearRegression,

        "decision_tree": DecisionTreeRegressor,

        "random_forest": RandomForestRegressor
    }

    if model_name not in models:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {list(models)}"
        )

    if model_name == "linear_regression":
        return LinearRegression(**model_params)

    if model_name == "decision_tree":
        model_params.setdefault(
            "random_state", random_state
        )

        return DecisionTreeRegressor(**model_params)

    if model_name == "random_forest":
        model_params.setdefault(
            "n_estimators", 300
        )

        model_params.setdefault(
            "random_state", random_state
        )

        model_params.setdefault(
            "n_jobs", -1
        )

        return RandomForestRegressor(**model_params)


# ============================================================
# 5. BUILD COMPLETE GENERIC REGRESSION PIPELINE
# ============================================================

def build_generic_pipeline(
    X_train,
    model_name="random_forest",
    categorical_columns=None,
    model_params=None,
    random_state=42,
    target_transform=None
):
    """
    Combine automatic preprocessing and regression model.

    Optionally transform a strictly positive target with a logarithm.
    The returned pipeline can be fitted, saved and reused.
    """

    preprocessor = build_generic_preprocessor(
        X_train,
        categorical_columns=categorical_columns
    )

    model = create_regression_model(
        model_name=model_name,
        random_state=random_state,
        model_params=model_params
    )

    if target_transform == "log":
        model = TransformedTargetRegressor(
            regressor=model,
            func=np.log,
            inverse_func=np.exp,
        )
    elif target_transform is not None:
        raise ValueError(
            f"Unsupported target transform: {target_transform}"
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
