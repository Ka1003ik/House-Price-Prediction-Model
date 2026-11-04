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
MODEL_DIR = PROJECT_ROOT / "models"

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


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Universal House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

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
      <a href="#12-house-price-prediction">12&nbsp; · &nbsp;Price prediction</a>
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
    st.info("Upload a CSV or Excel dataset to access Sections 2–11.")
else:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        dataset_signature = hashlib.sha256(uploaded_bytes).hexdigest()

        df, is_ames_dataset = load_uploaded_dataset(uploaded_file)

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")
        if not df.columns.is_unique:
            raise ValueError("Dataset contains duplicate column names.")

        # Do not keep in-session training or prediction results from a
        # different dataset used by the training workflow. The persisted
        # best model on disk is intentionally not deleted here.
        if st.session_state.get("_dataset_signature") != dataset_signature:
            clear_training_state()
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

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Numerical Features")
        st.metric("Detected Features", len(numeric_columns))

        with st.expander(
            f"View all {len(numeric_columns)} numerical features"
        ):
            st.dataframe(
                pd.DataFrame({"Feature": numeric_columns}),
                use_container_width=True,
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
                use_container_width=True,
                hide_index=True,
            )

    st.header("5. Configuration Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric("Target Variable", target_column)
    summary_col2.metric("Input Features", len(feature_columns))
    summary_col3.metric("Excluded ID Columns", len(id_columns))

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
        st.pyplot(fig, use_container_width=True)
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

        finite_target = target_numeric[
            np.isfinite(target_numeric_array)
        ]

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        stat_col1.metric(
            "Average",
            f"{finite_target.mean():,.2f}"
        )

        stat_col2.metric(
            "Median",
            f"{finite_target.median():,.2f}"
        )

        stat_col3.metric(
            "Minimum",
            f"{finite_target.min():,.2f}"
        )

        stat_col4.metric(
            "Maximum",
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
        st.pyplot(fig, use_container_width=True)
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
                "target_column": target_column
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

        metric_labels = {
            "Validation_RMSE": "Validation RMSE",
            "Validation_MAE": "Validation MAE",
            "Validation_MSE": "Validation MSE",
            "Validation_R2": "Validation R²",
            "CV_RMSE_Mean": "Cross-validation RMSE",
        }
        metric_choice = st.selectbox(
            "Select evaluation metric",
            options=list(metric_labels),
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
        st.pyplot(fig, use_container_width=True)
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
            "CV_RMSE_Mean": "CV RMSE (Mean)",
            "Train_RMSE": "Training RMSE",
            "Validation_RMSE": "Validation RMSE",
            "Validation_MAE": "Validation MAE",
            "Validation_R2": "Validation R²",
        })
        st.dataframe(
            comparison_display.round(3),
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
            ax.set_title(f"Actual vs Predicted House Prices — {model_display_names.get(selected_model, selected_model)}", pad=14)
            ax.set_xlabel("Actual House Price (₹)")
            ax.set_ylabel("Predicted House Price (₹)")
            ax.ticklabel_format(style="plain", axis="both")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            display_predictions = prediction_df.head(20).copy()
            display_predictions = display_predictions.rename(columns={
                "Actual Price": "Actual Price (₹)",
                "Predicted Price": "Predicted Price (₹)",
            })
            st.dataframe(
                display_predictions.round(2),
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
            ax.set_xlabel("Residual (Actual − Predicted) (₹)")
            ax.set_ylabel("Number of Homes")
            ax.set_title(f"Residual Distribution — {model_display_names.get(selected_model, selected_model)}")
            ax.ticklabel_format(style="plain", axis="x")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
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

        st.success(f"Best performing model: {model_display_names.get(best_result['Model'], best_result['Model'])}")

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
                "Residual": "Prediction_Error",
                "Absolute_Error": "Absolute_Error",
            })
            evaluation_display = evaluation_df.head(10).rename(columns={
                "Actual_Price": "Actual Price (₹)",
                "Predicted_Price": "Predicted Price (₹)",
                "Prediction_Error": "Prediction Error (₹)",
                "Absolute_Error": "Absolute Error (₹)",
            })
            st.dataframe(
                evaluation_display.round(2),
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

                    st.success("Feature importance calculated successfully!")

                except Exception as error:
                    st.error(f"Feature importance calculation failed: {error}")

        if "feature_importance" in st.session_state:

            importance_df = st.session_state["feature_importance"]

            st.subheader("Feature Importance Ranking")

            st.dataframe(
                importance_df.round(3),
                use_container_width=True,
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
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


        # --------------------------------------------------------
        # 11.2 SHAP EXPLAINABILITY
        # --------------------------------------------------------

        st.subheader("11.2 SHAP Explainability")

        st.write(
            "Understand how individual features contribute "
            "to the model's house-price predictions."
        )

        if st.button("Calculate SHAP Explanations"):

            with st.spinner("Calculating SHAP values..."):

                try:
                    shap_values = calculate_shap_values(
                        pipeline=best_pipeline,
                        X_background=X_valid,
                        X_explain=X_valid
                    )

                    st.session_state["shap_values"] = shap_values

                    st.success("SHAP explanations calculated successfully!")

                except Exception as error:
                    st.error(f"SHAP calculation failed: {error}")

        if "shap_values" in st.session_state:

            shap_values = st.session_state["shap_values"]

            # Global SHAP importance
            st.markdown("#### Global SHAP Feature Importance")

            shap_importance = pd.DataFrame({
                "Feature": shap_values.feature_names,
                "Mean_Absolute_SHAP": np.abs(
                    shap_values.values
                ).mean(axis=0)
            })

            shap_importance = shap_importance.sort_values(
                by="Mean_Absolute_SHAP",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                shap_importance.head(15).round(3),
                use_container_width=True,
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
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_ylabel("Feature")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Individual prediction explanation
            st.markdown("#### Individual House Prediction Explanation")

            selected_row = st.selectbox(
                "Select a house record to explain",
                options=list(range(len(shap_values.values))),
                format_func=lambda x: f"House Record {x + 1}"
            )

            local_explanation = pd.DataFrame({
                "Feature": shap_values.feature_names,
                "SHAP_Value": shap_values.values[selected_row]
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

            st.dataframe(
                local_explanation.round(3),
                use_container_width=True,
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
            ax.set_title("How Features Influence This House Price Prediction")
            ax.set_xlabel("SHAP Value")
            ax.set_ylabel("Feature")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# --------------------------------------------------------
# 12. HOUSE PRICE PREDICTION INTERFACE
# --------------------------------------------------------

st.header("12. House Price Prediction")

st.write(
    "Upload a housing dataset to generate price predictions using the "
    "best previously saved machine learning model. You do not need to "
    "retrain a model when a valid trained model is already available."
)

metadata_path = MODEL_DIR / "best_model_metadata.json"
saved_model_path = None
saved_feature_columns = None
saved_pipeline = None
saved_model_signature = None
model_validation_error = None

# Access is based on the persisted model artifacts, not Streamlit session
# state. This keeps prediction available after an application restart.
try:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Saved model metadata was not found. Complete model training "
            "in Section 8 first."
        )

    metadata_bytes = metadata_path.read_bytes()
    saved_metadata = json.loads(metadata_bytes.decode("utf-8"))

    if not isinstance(saved_metadata, dict):
        raise ValueError("Saved model metadata must be a JSON object.")

    model_file = saved_metadata.get("model_file")
    saved_feature_columns = saved_metadata.get("feature_columns")
    best_model_label = saved_metadata.get("best_model_name")
    target_label = saved_metadata.get("target_column")

    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("Model metadata does not contain a valid model file path.")

    if (
        not isinstance(saved_feature_columns, list)
        or not saved_feature_columns
        or any(not isinstance(column, str) or not column.strip()
               for column in saved_feature_columns)
        or len(saved_feature_columns) != len(set(saved_feature_columns))
    ):
        raise ValueError("Model metadata contains an invalid feature-column schema.")

    if not isinstance(best_model_label, str) or not best_model_label.strip():
        raise ValueError("Model metadata does not identify the best trained model.")

    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError("Model metadata does not identify the training target column.")

    candidate_model_path = Path(model_file).expanduser()
    if not candidate_model_path.is_absolute():
        candidate_model_path = PROJECT_ROOT / candidate_model_path
    candidate_model_path = candidate_model_path.resolve()

    # A metadata path must resolve inside this project's folder.
    if not candidate_model_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Saved model path points outside the project directory.")

    if not candidate_model_path.is_file():
        raise FileNotFoundError(
            f"Saved model file was not found: {candidate_model_path.name}"
        )

    # Deserializing catches corrupt/incompatible joblib files before the
    # upload control is enabled. The prediction method is checked as well.
    saved_pipeline = load_prediction_pipeline(candidate_model_path)
    if not callable(getattr(saved_pipeline, "predict", None)):
        raise ValueError("The saved artifact is not a usable prediction model.")

    # Confirm that the saved metadata schema matches the fitted pipeline.
    # Use a non-existent identifier name so a feature named 'Id' is not
    # accidentally discarded during this schema-only check.
    schema_probe = prepare_prediction_features(
        pd.DataFrame(columns=saved_feature_columns),
        saved_pipeline,
        id_column="__estate_ai_schema_check_only__",
    )
    if list(schema_probe.columns) != saved_feature_columns:
        raise ValueError(
            "Saved model metadata does not match the model's fitted feature schema."
        )

    saved_model_path = candidate_model_path
    model_stat = saved_model_path.stat()
    saved_model_signature = hashlib.sha256(
        metadata_bytes
        + f"{saved_model_path}:{model_stat.st_size}:{model_stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()

except Exception as error:
    model_validation_error = str(error)
    saved_model_path = None
    saved_feature_columns = None
    saved_pipeline = None
    saved_model_signature = None

if saved_pipeline is None:
    st.warning(
        "**Prediction Access Restricted**\n\n"
        "To generate house price predictions, you must first train a "
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
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Generate Predictions")
            if st.button(
                "Generate House Price Predictions",
                type="primary",
                use_container_width=True,
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
                            prediction_results["Predicted_House_Price"] = predictions

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
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Predictions", f"{len(predictions):,}")
                    col2.metric("Average Predicted Price", f"{np.mean(predictions):,.2f}")
                    col3.metric("Minimum Predicted Price", f"{np.min(predictions):,.2f}")
                    col4.metric("Maximum Predicted Price", f"{np.max(predictions):,.2f}")

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
                        label=f"Average: {np.mean(predictions):,.0f}",
                    )
                    ax.axvline(
                        np.median(predictions),
                        color="#112B46",
                        linestyle=":",
                        linewidth=2,
                        label=f"Median: {np.median(predictions):,.0f}",
                    )
                    ax.set_xlabel("Predicted House Price")
                    ax.set_ylabel("Number of Houses")
                    ax.set_title("Distribution of Predicted House Prices")
                    ax.legend()
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(axis="y", alpha=0.2)
                    ax.set_axisbelow(True)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.markdown("### Prediction Results")
                    st.dataframe(
                        prediction_results,
                        use_container_width=True,
                        hide_index=True,
                    )
                    csv_output = prediction_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_output,
                        file_name="house_price_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
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