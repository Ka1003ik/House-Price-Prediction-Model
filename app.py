import hashlib
import io
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


def clear_training_state():
    """Remove cached model results when the dataset/configuration changes."""
    keys_to_clear = [
        "training_results",
        "trained_models",
        "validation_actual",
        "validation_features",
        "best_model_name",
        "best_model_path",
        "_training_config_signature",
        "actual_predicted_model",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Universal House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Universal House Price Prediction System")
st.write("Upload, inspect and prepare datasets for machine learning.")


# ==================================================
# 2. DATASET UPLOAD
# ==================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset to get started.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    dataset_signature = hashlib.sha256(uploaded_bytes).hexdigest()

    df, is_ames_dataset = load_uploaded_dataset(uploaded_file)

    if df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    if not df.columns.is_unique:
        st.error("Dataset contains duplicate column names.")
        st.stop()

    # Do not keep predictions from a different uploaded dataset.
    if st.session_state.get("_dataset_signature") != dataset_signature:
        clear_training_state()
        st.session_state["_dataset_signature"] = dataset_signature

    st.success("Dataset uploaded successfully!")
    if is_ames_dataset:
        st.caption(
            "Ames Housing format detected. Known absence categories "
            "such as no garage/no pool are preserved separately from "
            "genuinely missing values."
        )

except Exception as error:
    st.error(f"Unable to read dataset: {error}")
    st.stop()


# ==================================================
# 3. DATASET OVERVIEW
# ==================================================

st.header("2. Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)


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


# ==================================================
# 5. AUTOMATIC FEATURE DETECTION AND CONFIGURATION
# ==================================================

st.header("4. Automatic Feature Detection")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical Features")
    st.metric("Count", len(numeric_columns))
    st.write(numeric_columns)

with col2:
    st.subheader("Categorical Features")
    st.metric("Count", len(categorical_columns))
    st.write(categorical_columns)

st.header("5. Configuration Summary")
st.write("**Target column:**", target_column)
st.write("**ID columns:**", id_columns or "None")
st.write("**Input features:**", len(feature_columns))
st.write("**Numerical features:**", len(numeric_columns))
st.write("**Categorical features:**", len(categorical_columns))
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
    use_container_width=True,
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
    st.bar_chart(
        missing_columns.set_index("Column")["Missing Values"]
    )

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
    use_container_width=True,
    hide_index=True,
)

st.subheader("6.4 Target Column Validation")
st.write("Selected target:", target_column)

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
    finite_target = target_numeric[np.isfinite(target_numeric_array)]
    st.write(finite_target.describe())


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
)

cv_folds = st.selectbox(
    "Cross-validation folds",
    options=[3, 5],
    index=1,
)

random_state = st.number_input(
    "Random state",
    min_value=0,
    value=42,
    step=1,
)

st.subheader("Select Regression Models")
selected_models = st.multiselect(
    "Choose models to train",
    options=[
        "linear_regression",
        "decision_tree",
        "random_forest",
    ],
    default=[
        "linear_regression",
        "decision_tree",
        "random_forest",
    ],
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    and st.session_state.get("_training_config_signature")
    != current_training_signature
):
    clear_training_state()
    st.info(
        "Your dataset or training configuration has changed. "
        "Train the models again to refresh the results."
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

        st.success(
            f"Best model saved successfully: {best_model_name}"
        )
        st.info(f"Model location: {best_model_path}")
        st.success("All selected models trained successfully!")

    except Exception as error:
        st.error(f"Model training failed: {error}")


# ==================================================
# 9. TRAINING RESULTS
# ==================================================

if "training_results" in st.session_state:
    st.header("9. Model Training Results")

    results_df = st.session_state["training_results"]

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    best_model_name = st.session_state["best_model_name"]
    best_model = results_df.loc[
        results_df["Model"] == best_model_name
    ].iloc[0]

    st.subheader("Best Performing Model")
    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best_model["Model"])
    col2.metric(
        "Validation RMSE",
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

    metric_choice = st.selectbox(
        "Select evaluation metric",
        options=[
            "Validation_RMSE",
            "Validation_MAE",
            "Validation_MSE",
            "Validation_R2",
            "CV_RMSE_Mean",
        ],
    )

    chart_data = results_df.set_index("Model")[[metric_choice]]
    st.bar_chart(chart_data)
    st.caption(
        "For RMSE, MAE and MSE, lower values are better. "
        "For R², higher values are better."
    )

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

    st.dataframe(
        results_df[comparison_columns].round(4),
        use_container_width=True,
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

        st.scatter_chart(
            prediction_df,
            x="Actual Price",
            y="Predicted Price",
        )

        st.dataframe(
            prediction_df.head(20).round(2),
            use_container_width=True,
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
        col1.metric("Mean Residual", f"{np.mean(residuals):,.2f}")
        col2.metric("Minimum Residual", f"{np.min(residuals):,.2f}")
        col3.metric("Maximum Residual", f"{np.max(residuals):,.2f}")

        st.write("Residual Distribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(residuals, bins=30)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Number of Homes")
        ax.set_title(f"Residual Distribution — {selected_model}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("Residual Outlier Investigation")
        residual_check = pd.DataFrame({
            "Actual": actual_values,
            "Predicted": predicted_values,
            "Residual": residuals,
            "Absolute_Error": np.abs(residuals),
        })

        st.dataframe(
            residual_check.sort_values(
                "Absolute_Error", ascending=False
            ).head(10).round(2),
            use_container_width=True,
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

    st.success(f"Best performing model: {best_result['Model']}")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Validation RMSE",
        f"{best_result['Validation_RMSE']:,.2f}",
    )
    col2.metric(
        "Validation MAE",
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
        })

        st.dataframe(
            evaluation_df.head(10).round(2),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = evaluation_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Evaluation Results",
            data=csv_data,
            file_name=f"house_price_evaluation_{selected_model}.csv",
            mime="text/csv",
        )

# ============================================================
# 11. HOUSE PRICE PREDICTION INTERFACE
# ============================================================

st.header("11. House Price Prediction Interface")

st.write(
    "Upload a CSV file containing house features to generate "
    "predictions using the saved best-performing model."
)

if (
    "best_model_path" in st.session_state
    and "feature_columns" in st.session_state
):

    uploaded_prediction_file = st.file_uploader(
        "Upload House Dataset for Prediction",
        type=["csv"],
        key="prediction_upload"
    )

    if uploaded_prediction_file is not None:

        prediction_df = pd.read_csv(uploaded_prediction_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(prediction_df.head())

        expected_columns = st.session_state["feature_columns"]

        # Remove common ID columns if present
        prediction_input = prediction_df.drop(
            columns=["Id"],
            errors="ignore"
        )

        missing_columns = [
            col for col in expected_columns
            if col not in prediction_input.columns
        ]

        extra_columns = [
            col for col in prediction_input.columns
            if col not in expected_columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        elif extra_columns:

            st.warning(
                f"Extra columns will be ignored: {extra_columns}"
            )

        else:

            prediction_input = prediction_input[expected_columns]

            if st.button("Generate House Price Predictions"):

                best_pipeline = joblib.load(
                    st.session_state["best_model_path"]
                )

                predictions = best_pipeline.predict(
                    prediction_input
                )

                prediction_results = prediction_df.copy().reset_index(drop=True)

                prediction_results["Predicted_House_Price"] = (
                    predictions
                )

                st.success("Predictions generated successfully!")

                st.subheader("Prediction Results")

                st.dataframe(
                    prediction_results,
                    use_container_width=True
                )

                st.metric(
                    "Average Predicted House Price",
                    f"{np.mean(predictions):,.2f}"
                )

                csv_output = prediction_results.to_csv(index=False)

                st.download_button(
                    label="Download House Price Predictions",
                    data=csv_output,
                    file_name="house_price_predictions.csv",
                    mime="text/csv"
                )

else:

    st.warning(
        "Please train the models first to generate predictions."
    )