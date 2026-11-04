
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

ABSENCE_COLUMNS = [
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
    "MiscFeature"
]


NUMERIC_COLUMNS = [
    "Id",
    "MSSubClass",
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SalePrice"
]


# Numeric codes that represent categories rather than quantities.
CATEGORICAL_NUMERIC_COLUMNS = [
    "MSSubClass"
]


# ============================================================
# 2. LOAD AND NORMALIZE AMES DATASET
# ============================================================

def load_ames_dataset(file_path):
    """
    Load the Ames dataset and preserve meaningful
    categories representing the absence of features.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        keep_default_na=False,
        na_values=[""]
    )

    # Convert documented absence labels into an explicit category.
    for column in ABSENCE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].replace("NA", "None")

    # MasVnrType uses None for no veneer.
    # NA is treated as genuinely missing information.
    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].replace(
            "NA", np.nan
        )

    # Convert numerical columns safely.
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# 3. NEIGHBORHOOD MEDIAN IMPUTER
# ============================================================

class NeighborhoodMedianImputer(
    BaseEstimator,
    TransformerMixin
):
    """
    Fill missing LotFrontage using neighborhood medians.
    Use the global training median as a fallback.
    """

    def __init__(
        self,
        target_col="LotFrontage",
        group_col="Neighborhood"
    ):
        self.target_col = target_col
        self.group_col = group_col

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )
        # Store the original feature names and count.
            self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object
)
        self.n_features_in_ = X.shape[1]

        required_columns = [
            self.target_col,
            self.group_col
        ]

        missing_columns = [
            col for col in required_columns
            if col not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        frontage = pd.to_numeric(
            X[self.target_col],
            errors="coerce"
        )

        neighborhoods = (
            X[self.group_col]
            .astype("string")
            .fillna("__UNKNOWN_NEIGHBORHOOD__")
        )

        self.global_median_ = frontage.median()

        if pd.isna(self.global_median_):
            raise ValueError(
                "Cannot calculate the global frontage median."
            )

        self.neighborhood_medians_ = (
            frontage.groupby(neighborhoods)
            .median()
            .to_dict()
        )

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if not hasattr(self, "global_median_"):
            raise RuntimeError(
                "Fit the imputer before transforming data."
            )

        required_columns = [
            self.target_col,
            self.group_col
        ]

        missing_columns = [
            col for col in required_columns
            if col not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for imputation: {missing_columns}"
            )

        result = X.copy()

        frontage = pd.to_numeric(
            result[self.target_col],
            errors="coerce"
        )

        neighborhoods = (
            result[self.group_col]
            .astype("string")
            .fillna("__UNKNOWN_NEIGHBORHOOD__")
        )

        neighborhood_medians = neighborhoods.map(
            self.neighborhood_medians_
        )

        result[self.target_col] = (
            frontage
            .fillna(neighborhood_medians)
            .fillna(self.global_median_)
        )

        return result


# ============================================================
# 4. BUILD REUSABLE PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X_train):
    """
    Automatically identify numerical and categorical features
    and create a reusable preprocessing pipeline.
    """

    numeric_features = [
        col
        for col in X_train.select_dtypes(
            include=["number"]
        ).columns
        if col not in CATEGORICAL_NUMERIC_COLUMNS
    ]

    categorical_features = [
        col
        for col in X_train.columns
        if col not in numeric_features
    ]

    # Numerical feature processing.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            )
        ]
    )

    # Categorical feature processing.
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

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numerical",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    column_preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    steps = []

    # Apply neighborhood-specific imputation first.
    if (
        "LotFrontage" in X_train.columns
        and "Neighborhood" in X_train.columns
    ):
        steps.append(
            (
                "neighborhood_imputer",
                NeighborhoodMedianImputer()
            )
        )

    steps.append(
        (
            "column_preprocessor",
            column_preprocessor
        )
    )

    return Pipeline(steps=steps)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

def separate_features_target(
    df,
    target_column="SalePrice",
    id_columns=("Id",)
):
    """
    Separate input features (X) and target (y).
    Exclude identifier columns from model training.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    y = pd.to_numeric(
        df[target_column],
        errors="raise"
    )

    columns_to_remove = [
        target_column,
        *id_columns
    ]

    X = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    ).copy()

    return X, y

