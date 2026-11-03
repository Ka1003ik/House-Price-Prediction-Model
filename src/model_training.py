
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. CREATE REGRESSION MODELS
# ============================================================

def create_regression_models(random_state=42):
    """
    Create the three regression models required
    for the house price prediction project.
    """

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regression": DecisionTreeRegressor(
            random_state=random_state
        ),

        "Random Forest Regression": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        )
    }

    return models


# ============================================================
# 2. EVALUATE REGRESSION MODEL
# ============================================================

def evaluate_model(y_actual, y_predicted):
    """
    Calculate regression evaluation metrics.
    """

    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ============================================================
# 3. TRAIN AND COMPARE MODELS
# ============================================================

def train_and_evaluate_models(
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_state=42
):
    """
    Train all regression models and evaluate their
    performance on validation data.

    Returns:
        results: Model comparison DataFrame
        trained_models: Dictionary of fitted models
        predictions: Dictionary of validation predictions
    """

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training features and target have different row counts."
        )

    if X_valid.shape[0] != len(y_valid):
        raise ValueError(
            "Validation features and target have different row counts."
        )

    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )

    models = create_regression_models(random_state)

    results = []
    trained_models = {}
    predictions = {}

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        # Train the model.
        model.fit(X_train, y_train)

        # Predict validation house prices.
        y_predicted = model.predict(X_valid)

        # Calculate evaluation metrics.
        metrics = evaluate_model(
            y_valid,
            y_predicted
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        trained_models[model_name] = model
        predictions[model_name] = y_predicted

        print("Training completed.")

    results_df = pd.DataFrame(results)

    # Lower MSE is better.
    results_df = results_df.sort_values(
        by="MSE",
        ascending=True
    ).reset_index(drop=True)

    return results_df, trained_models, predictions
