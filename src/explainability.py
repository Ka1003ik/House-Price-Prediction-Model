import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def calculate_permutation_importance(
    pipeline,
    X_valid,
    y_valid,
    n_repeats=5,
    random_state=42
):
    
    """
    Calculate feature importance using permutation importance.

    Works with the complete fitted preprocessing pipeline.
    Returns importance using original feature names.
    """

    if X_valid.empty:
        raise ValueError("Validation dataset is empty.")

    if len(X_valid) != len(y_valid):
        raise ValueError(
            "Features and target have different row counts."
        )

    result = permutation_importance(
        pipeline,
        X_valid,
        y_valid,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "Feature": X_valid.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        by="Importance_Mean",
        ascending=False
    ).reset_index(drop=True)

    return importance_df


def calculate_shap_values(
    pipeline,
    X_background,
    X_explain,
    max_evals=None
):
    """
    Calculate SHAP values using transformed numerical features.
    Supports Linear Regression, Decision Tree and Random Forest.
    """

    if X_background.empty or X_explain.empty:
        raise ValueError("Input datasets cannot be empty.")

    if not X_background.columns.equals(X_explain.columns):
        raise ValueError(
            "Background and explanation features must match."
        )

    background = X_background.sample(
        n=min(20, len(X_background)),
        random_state=42
    )

    explain_data = X_explain.head(5).copy()

    # Extract fitted preprocessing and prediction model
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    # Transform raw features into numerical features
    background_transformed = preprocessor.transform(background)
    explain_transformed = preprocessor.transform(explain_data)

    # Convert sparse matrices to dense numerical arrays
    if sparse.issparse(background_transformed):
        background_transformed = background_transformed.toarray()

    if sparse.issparse(explain_transformed):
        explain_transformed = explain_transformed.toarray()

    background_transformed = np.asarray(background_transformed)
    explain_transformed = np.asarray(explain_transformed)

    feature_names = list(preprocessor.get_feature_names_out())

    # Select the appropriate SHAP explainer
    if isinstance(
        estimator,
        (RandomForestRegressor, DecisionTreeRegressor)
    ):
        explainer = shap.TreeExplainer(
            estimator,
            data=background_transformed,
            feature_perturbation="interventional"
        )

        values = explainer.shap_values(explain_transformed)

    elif isinstance(estimator, LinearRegression):
        explainer = shap.LinearExplainer(
            estimator,
            background_transformed
        )

        values = explainer.shap_values(explain_transformed)

    else:
        explainer = shap.Explainer(
            estimator.predict,
            background_transformed,
            feature_names=feature_names
        )

        if max_evals is None:
            return explainer(explain_transformed)

        return explainer(
            explain_transformed,
            max_evals=max_evals
        )

    values = np.asarray(values)

    base_values = np.full(
        values.shape[0],
        float(np.asarray(explainer.expected_value).squeeze())
    )

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=explain_transformed,
        feature_names=feature_names
    )
