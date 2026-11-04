
# Universal House Price Prediction System

An end-to-end machine learning project for house price prediction, covering data preprocessing, model training, hyperparameter tuning, model explainability, and interactive predictions through a Streamlit web application.

## Overview

The Universal House Price Prediction System is designed to develop and evaluate machine learning models for predicting residential property prices. It includes a reusable preprocessing and modelling pipeline, model evaluation utilities, explainability features, and an interactive application for generating predictions.

The project uses the Ames Housing dataset as its primary dataset and also includes a UCI Real Estate Valuation dataset for additional experimentation.

## Key Features

- **Data Preprocessing:** Handles missing values, categorical variables, numerical features, and feature transformations.
- **Model Training:** Supports multiple regression algorithms for house price prediction.
- **Model Comparison:** Evaluates models using regression performance metrics.
- **Hyperparameter Tuning:** Improves model performance through parameter optimization.
- **Model Explainability:** Provides feature importance and SHAP-based explanations.
- **Prediction Validation:** Validates prediction inputs and handles invalid numeric values.
- **Interactive Web Application:** Uses Streamlit to provide an interface for exploring the system and generating predictions.
- **Reusable Pipeline:** Separates preprocessing, training, tuning, prediction, and explainability into modular Python components.

## Technologies Used

| Category             | Technologies        |
|----------------------|---------------------|
| Programming Language | Python              |
| Data Analysis        | Pandas, NumPy       |
| Machine Learning     | Scikit-learn        |
| Model Explainability | SHAP                |
| Data Visualization   | Matplotlib, Seaborn |
| Model Serialization  | Joblib              |
| Web Application      | Streamlit           |
| Excel Support        | OpenPyXL, xlrd      |

## Project Structure

```text
LEVEL_2_INTERMEDIATE/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   ├── sample_submission.csv
│   └── uci_real_estate/
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

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd LEVEL_2_INTERMEDIATE
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default browser. If it does not, use the local URL displayed in the terminal.

## Model Evaluation

The project uses standard regression evaluation metrics, including:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

These metrics help evaluate prediction accuracy and compare different regression models.

## Datasets

**Ames Housing Dataset**

The primary dataset is from the Kaggle House Prices: Advanced Regression Techniques competition.

Dataset source: [Kaggle – House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

**UCI Real Estate Valuation Dataset**

An additional real estate valuation dataset is included for experimentation.

Dataset source: [UCI – Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)

## Project Objectives

1. Develop a structured machine learning workflow for house price prediction.
2. Implement reusable data preprocessing and model training pipelines.
3. Compare and optimize regression models.
4. Interpret model predictions using explainability techniques.
5. Build an interactive application for practical prediction.

## Author

**Kaushik K A**

B.Tech – Artificial Intelligence and Data Science

GitHub: [Ka1003ik](https://github.com/Ka1003ik)
