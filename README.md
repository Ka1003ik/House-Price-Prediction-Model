
# House Price Prediction Model

An end-to-end machine learning project for training, evaluating, explaining, and serving house-price regression models. The Streamlit application supports individual and bulk predictions using saved models, while the accompanying Python modules and notebooks provide reusable data-processing and modelling workflows.

**Repository:** [github.com/Ka1003ik/House-Price-Prediction-Model](https://github.com/Ka1003ik/House-Price-Prediction-Model)

## Features

- Data inspection, cleaning, preprocessing, missing-value handling, and categorical encoding.
- Train and compare Linear Regression, Decision Tree, and Random Forest models.
- Evaluate models with cross-validation and regression metrics (RMSE, MAE, MSE, and R²).
- Tune model hyperparameters and inspect feature importance and SHAP explanations.
- Generate individual predictions or upload CSV/Excel files for bulk predictions.
- Validate prediction inputs against the saved model's feature schema.
- Save trained pipelines and metadata for reuse between application sessions.
- Work with the Ames Housing and UCI Real Estate Valuation datasets.

## Technologies

Python, Streamlit, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Seaborn, Joblib, and OpenPyXL.

## Repository Structure

```text
House-Price-Prediction-Model/
├── .streamlit/                 # Streamlit configuration
├── app.py                      # Interactive Streamlit application
├── dataset/                    # Ames and UCI datasets and descriptions
├── models/                     # Saved model pipelines, metadata, and artifacts
│   └── datasets/               # Dataset-specific model workspaces
├── notebooks/                  # Numbered analysis and modelling workflows
├── outputs/                    # Evaluation tables, predictions, and plots
├── src/                        # Reusable preprocessing, modelling, and prediction code
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### Clone the repository

```bash
git clone https://github.com/Ka1003ik/House-Price-Prediction-Model.git
cd House-Price-Prediction-Model
```

### Create and activate a virtual environment

Create the environment:

```bash
python -m venv .venv
```

Activate it in **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it in **Windows Command Prompt**:

```bat
.venv\Scripts\activate.bat
```

Activate it in **Linux or macOS**:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the application

```bash
streamlit run app.py
```

Streamlit prints the local application URL in the terminal and normally opens it in your browser.

## Using the Application

1. Upload a dataset containing a numeric target column.
2. Select the target, any identifier columns to exclude, and the models to train.
3. Configure the validation split and cross-validation settings, then train.
4. Review the model comparison, validation metrics, and feature explanations.
5. Use the saved model to predict an individual property or upload a CSV/Excel file for batch predictions.

Prediction files must include the features required by the selected saved model. The application reports missing or incompatible columns before generating predictions. Target labels and units are read from the saved model metadata; for the UCI dataset, the target is house price per unit area in 10,000 TWD / Ping.

## Datasets

- **Ames Housing:** Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
- **Real Estate Valuation:** UCI [Real Estate Valuation Data Set](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set).

## Notes

- Model files are serialized with Joblib. Use compatible Python and scikit-learn versions when loading saved models; dependency versions are listed in `requirements.txt`.
- Training a model in the application saves model files and metadata in the corresponding `models/` workspace.
- Generated charts, metrics, and prediction outputs are stored in `outputs/` by the analysis workflows.

## Author

**Kaushik K A**

B.Tech, Artificial Intelligence and Data Science

GitHub: [Ka1003ik](https://github.com/Ka1003ik)
