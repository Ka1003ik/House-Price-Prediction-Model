# Universal House Price Prediction System

An end-to-end machine learning project for house price prediction, featuring data preprocessing, model training, hyperparameter tuning, model evaluation, explainability, and interactive predictions through a Streamlit web application.

## Overview

The Universal House Price Prediction System is a machine learning application designed to develop, evaluate, and interpret regression models for predicting residential property prices.

The project implements a reusable preprocessing and modelling pipeline, supports multiple regression algorithms, provides model evaluation and explainability tools, and offers an interactive interface for dataset exploration and price prediction.

The primary project workflow uses the Ames Housing dataset, while a separate feature-ablation experiment evaluates the effect of excluding a predictor from the Boston Housing dataset.

## Key Features

- **Data Preprocessing:** Handles missing values, categorical variables, numerical features, and feature transformations.
- **Automatic Feature Detection:** Identifies numerical and categorical input features.
- **Data Quality Analysis:** Examines missing values, duplicate records, and target validity.
- **Model Training:** Supports multiple regression algorithms.
- **Model Comparison:** Evaluates models using standard regression metrics.
- **Hyperparameter Tuning:** Supports parameter optimization for model performance improvement.
- **Model Explainability:** Uses permutation feature importance and SHAP-based explanations.
- **Prediction Validation:** Validates prediction inputs and handles invalid numerical values.
- **Interactive Web Application:** Provides dataset exploration, model training, analytics, explainability, and prediction through Streamlit.
- **Reusable Pipeline:** Organizes preprocessing, training, tuning, prediction, and explainability into modular Python components.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Explainability | SHAP |
| Data Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Excel Support | OpenPyXL, xlrd |

## Project Structure

```text
House-Price-Prediction-Model/
│
├── app.py
├── requirements.txt
├── README.md
├── EXPERIMENT_COMPARISON.md
├── .gitignore
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
│
├── models/
│   ├── best_house_price_model.joblib
│   └── best_model_metadata.json
│
├── notebooks/
│   └── Data analysis and machine learning workflow scripts
│
├── outputs/
│   └── Evaluation results, visualizations, and prediction files
│
└── src/
    ├── preprocessing.py
    ├── generic_pipeline.py
    ├── model_training.py
    ├── model_tuning.py
    ├── prediction.py
    └── explainability.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ka1003ik/House-Price-Prediction-Model.git
```

### 2. Navigate to the Project Directory

```bash
cd House-Price-Prediction-Model
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default browser. If it does not, use the local URL displayed in the terminal.

## Model Evaluation

The project uses the following regression evaluation metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the target variable's original unit.
- **R-squared (R²):** Measures the proportion of target variance explained by the model.

These metrics provide a consistent basis for comparing regression model performance.

## Boston Housing Dataset: Feature Ablation Experiment

A separate experiment was conducted using the Boston Housing dataset to examine the effect of excluding `Feature_12 (B)` from the model inputs.

### Experimental Setup

- **Dataset records:** 506
- **Target variable:** `Feature_14` (Median House Value, Thousand USD)
- **Predictor features:** 13 (`Feature_1` through `Feature_13`)
- **Experiment 1:** All 13 input features included.
- **Experiment 2:** `Feature_12` excluded, retaining 12 input features.
- **Models evaluated:** Linear Regression, Decision Tree, and Random Forest.
- **Validation:** The same reported validation setup was used for both experiments.

### Best-Model Comparison

| Metric | With Feature_12 | Without Feature_12 |
|---|---:|---:|
| Best Model | Random Forest | Random Forest |
| Cross-validation RMSE | 3.8542 | 3.8647 |
| Validation RMSE | 2.9671 | 2.9495 |
| Validation MAE | 2.0674 | 2.0244 |
| Validation R² | 0.8799 | 0.8814 |

*RMSE and MAE are expressed in Thousand USD.*

### Experiment 2: Model-Level Validation Results

| Model | Validation RMSE | Validation MAE | Validation R² |
|---|---:|---:|---:|
| Linear Regression | 4.8144 | 3.2112 | 0.6839 |
| Decision Tree | 4.6083 | 2.6578 | 0.7104 |
| Random Forest | 2.9495 | 2.0244 | 0.8814 |

### Interpretation

Random Forest achieved the best validation performance in both experiments.

After excluding `Feature_12`, Random Forest validation RMSE decreased from 2.9671 to 2.9495 Thousand USD, while validation MAE decreased from 2.0674 to 2.0244 Thousand USD. Validation R² increased from 0.8799 to 0.8814.

Cross-validation RMSE increased slightly from 3.8542 to 3.8647 Thousand USD.

The differences are marginal and do not establish a statistically significant improvement. The results indicate that excluding `Feature_12` did not materially harm Random Forest validation performance in this experiment. The impact was model-dependent, as Decision Tree performance was worse in Experiment 2.

**Responsible Machine Learning Consideration:** The original Boston Housing variable `B` is derived from historical racial composition data. Excluding this variable is relevant to responsible machine learning. However, removing a single feature does not guarantee fairness or eliminate proxy discrimination.

For the detailed experiment comparison, refer to [EXPERIMENT_COMPARISON.md](EXPERIMENT_COMPARISON.md).

## Dataset Information

### Ames Housing Dataset

The primary dataset used in the project is the Ames Housing training dataset from the Kaggle House Prices: Advanced Regression Techniques competition.

- **Records:** 1,460
- **Total columns:** 81
- **Target variable:** `SalePrice`
- **Identifier column:** `Id`
- **Input features:** 79, excluding the identifier and target.

Dataset source: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

### Boston Housing Dataset

Used separately for the feature-ablation experiment described in this README.

- **Records:** 506
- **Predictor features:** 13
- **Target:** Median House Value (`MEDV`), represented in the application as `Feature_14`.

## Project Objectives

1. Develop a structured machine learning workflow for house price prediction.
2. Implement reusable preprocessing and model training pipelines.
3. Compare and optimize regression algorithms.
4. Evaluate model performance using appropriate metrics.
5. Interpret predictions using model explainability techniques.
6. Investigate the impact of feature exclusion on model performance.
7. Develop an interactive application for practical house price prediction.

## Author

**Kaushik K A**

B.Tech – Artificial Intelligence and Data Science

GitHub: [Ka1003ik](https://github.com/Ka1003ik)