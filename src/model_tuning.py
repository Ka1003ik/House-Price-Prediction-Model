
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.preprocessing import build_preprocessor


# ============================================================
# 1. MODEL EVALUATION
# ============================================================

def calculate_metrics(y_actual, y_predicted):

    mse = mean_squared_error(y_actual, y_predicted)

    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_actual, y_predicted),
        "R2": r2_score(y_actual, y_predicted)
    }


# ============================================================
# 2. CREATE COMPLETE PIPELINE
# ============================================================

def create_full_pipeline(X_train, model):

    """
    Combine preprocessing and regression model.

    During cross-validation, preprocessing is fitted
    separately inside each training fold.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ]
    )


# ============================================================
# 3. HYPERPARAMETER TUNING
# ============================================================

def tune_regression_models(
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_state=42,
    n_iter=10,
    cv_splits=5
):

    results = []
    trained_pipelines = {}
    tuning_details = {}

    # --------------------------------------------------------
    # LINEAR REGRESSION BASELINE
    # --------------------------------------------------------

    print("\nTraining Linear Regression baseline...")

    linear_pipeline = create_full_pipeline(
        X_train,
        LinearRegression()
    )

    linear_pipeline.fit(X_train, y_train)

    linear_train_pred = linear_pipeline.predict(X_train)
    linear_valid_pred = linear_pipeline.predict(X_valid)

    train_metrics = calculate_metrics(
        y_train,
        linear_train_pred
    )

    valid_metrics = calculate_metrics(
        y_valid,
        linear_valid_pred
    )

    results.append({
        "Model": "Linear Regression (Baseline)",
        "CV_RMSE": np.nan,
        "Train_RMSE": train_metrics["RMSE"],
        "Validation_RMSE": valid_metrics["RMSE"],
        "Validation_MAE": valid_metrics["MAE"],
        "Validation_R2": valid_metrics["R2"]
    })

    trained_pipelines["Linear Regression"] = linear_pipeline

    # --------------------------------------------------------
    # DECISION TREE SEARCH
    # --------------------------------------------------------

    print("\nTuning Decision Tree Regression...")

    decision_tree = DecisionTreeRegressor(
        random_state=random_state
    )

    decision_tree_pipeline = create_full_pipeline(
        X_train,
        decision_tree
    )

    decision_tree_params = {
        "model__max_depth": [
            None, 5, 8, 10, 12, 15, 20
        ],
        "model__min_samples_split": [
            2, 5, 10, 20
        ],
        "model__min_samples_leaf": [
            1, 2, 4, 6, 8
        ],
        "model__max_features": [
            None, 0.7, "sqrt"
        ]
    }

    # --------------------------------------------------------
    # RANDOM FOREST SEARCH
    # --------------------------------------------------------

    print("\nTuning Random Forest Regression...")

    random_forest = RandomForestRegressor(
        random_state=random_state,
        n_jobs=1
    )

    random_forest_pipeline = create_full_pipeline(
        X_train,
        random_forest
    )

    random_forest_params = {
        "model__n_estimators": [
            150, 250, 350
        ],
        "model__max_depth": [
            None, 10, 15, 20, 25
        ],
        "model__min_samples_split": [
            2, 5, 10
        ],
        "model__min_samples_leaf": [
            1, 2, 4
        ],
        "model__max_features": [
            1.0, 0.7, "sqrt"
        ]
    }

    # --------------------------------------------------------
    # CROSS-VALIDATION CONFIGURATION
    # --------------------------------------------------------

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    model_searches = [
        (
            "Decision Tree Regression",
            decision_tree_pipeline,
            decision_tree_params
        ),
        (
            "Random Forest Regression",
            random_forest_pipeline,
            random_forest_params
        )
    ]

    # --------------------------------------------------------
    # RUN RANDOMIZED SEARCH
    # --------------------------------------------------------

    for model_name, pipeline, parameters in model_searches:

        print(f"\nRunning {cv_splits}-fold cross-validation...")
        print(f"Testing {n_iter} parameter combinations.")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameters,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        # The best estimator is refitted on the full
        # training split automatically.
        best_pipeline = search.best_estimator_

        train_predictions = best_pipeline.predict(X_train)
        valid_predictions = best_pipeline.predict(X_valid)

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        valid_metrics = calculate_metrics(
            y_valid,
            valid_predictions
        )

        results.append({
            "Model": model_name + " (Tuned)",
            "CV_RMSE": -search.best_score_,
            "Train_RMSE": train_metrics["RMSE"],
            "Validation_RMSE": valid_metrics["RMSE"],
            "Validation_MAE": valid_metrics["MAE"],
            "Validation_R2": valid_metrics["R2"]
        })

        trained_pipelines[model_name] = best_pipeline

        tuning_details[model_name] = {
            "Best_CV_RMSE": -search.best_score_,
            "Best_Parameters": search.best_params_
        }

        print("\nBest parameters:")
        print(search.best_params_)

        print(
            f"Best cross-validation RMSE: "
            f"{-search.best_score_:,.2f}"
        )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Validation_RMSE"
    ).reset_index(drop=True)

    return results_df, trained_pipelines, tuning_details
