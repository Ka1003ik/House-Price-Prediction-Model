# 🏡 House Price Prediction System

An end-to-end **Machine Learning application** built with Python and Streamlit to analyze datasets, preprocess features, train and compare regression models, optimize hyperparameters, and generate house price predictions through an interactive interface.

The application is designed with a reusable machine learning pipeline that supports automated feature detection, model evaluation, explainability, and individual or batch predictions.

---

## 📌 Project Overview

House price prediction is a regression problem that involves estimating property values based on relevant features.

This project implements a complete machine learning workflow, covering:

- Dataset upload and exploration
- Automatic numerical and categorical feature detection
- Data preprocessing and feature transformation
- Regression model training and comparison
- Hyperparameter optimization
- Model evaluation and validation
- Feature importance and model explainability
- Individual and batch predictions

The application provides an interactive interface through Streamlit, making the machine learning workflow accessible without requiring users to execute individual Python scripts.

## 🚀 Live Demo

Experience the Universal House Price Prediction System:

🔗 [Launch Live Application](https://universal-house-price-predictor.streamlit.app/)

## 🎯 Project Objectives

- Develop a reusable end-to-end regression pipeline.
- Automate feature identification and preprocessing.
- Compare multiple regression algorithms.
- Improve model performance through hyperparameter tuning.
- Evaluate models using appropriate regression metrics.
- Interpret model predictions using explainability techniques.
- Provide an interactive prediction interface.

## 📊 Dataset

**Primary Dataset:** UCI Real Estate Valuation Dataset

**Source:** [UCI Machine Learning Repository – Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)

Dataset characteristics:

| Attribute | Details |
|---|---|
| Total records | 414 |
| Original columns | 8 |
| Predictor features | 6 |
| Target variable | House price of unit area |
| Problem type | Supervised Regression |

### Features

| Feature | Description |
|---|---|
| X1 | Transaction date |
| X2 | House age |
| X3 | Distance to the nearest MRT station |
| X4 | Number of convenience stores |
| X5 | Latitude |
| X6 | Longitude |
| Y | House price of unit area |

The dataset's record identifier is excluded from the predictor features.

## ⚙️ Machine Learning Workflow

### 1. Data Collection and Exploration
- Load datasets for analysis.
- Inspect dataset dimensions and structure.
- Examine data types and missing values.
- Explore numerical and categorical variables.
- Analyze target variable distribution.

### 2. Automatic Feature Detection
- Identify numerical features.
- Identify categorical features.
- Allow numerical columns to be treated as categorical when appropriate.
- Configure feature handling for preprocessing.

### 3. Data Preprocessing
- Handle missing values.
- Apply appropriate transformations to numerical features.
- Encode categorical variables.
- Prepare training and validation datasets.
- Build a reusable preprocessing pipeline.

### 4. Model Training

The application compares three regression algorithms:

| Algorithm | Model |
|---|---|
| Linear Regression | Baseline regression model |
| Decision Tree Regressor | Tree-based regression |
| Random Forest Regressor | Ensemble regression |

### 5. Hyperparameter Optimization

`RandomizedSearchCV` is used to optimize Decision Tree and Random Forest models.

The tuning process explores different combinations of model parameters using cross-validation and negative Root Mean Squared Error (RMSE) as the scoring metric.

The best-performing estimator is selected for subsequent evaluation.

### 6. Model Evaluation

The models are evaluated using:

- Cross-Validation RMSE
- Training RMSE
- Validation Mean Squared Error (MSE)
- Validation RMSE
- Validation R² Score

These metrics help compare model performance and assess generalization.

### 7. Model Explainability

The application incorporates model explainability techniques to understand feature contributions.

- Feature importance analysis
- SHAP-based explainability
- Interpretation of influential predictors

### 8. Prediction

The application supports:

- Individual house price prediction
- Batch predictions using uploaded input files
- Prediction outputs for further analysis

---

## 📈 Model Performance

The following results were obtained during hyperparameter tuning on the UCI Real Estate Valuation dataset.

### Cross-Validation Results

| Model | CV RMSE Mean | CV RMSE Std |
|---|---:|---:|
| Linear Regression | 9.1790 | 1.4189 |
| Decision Tree | 8.2600 | 1.5990 |
| Random Forest | **7.6014** | 1.5993 |

### Random Forest Validation Results

| Metric | Result |
|---|---:|
| Training RMSE | 4.7329 |
| Validation MSE | 27.8772 |
| Validation RMSE | 5.28 |
| Validation R² | 0.8338 |

**Observation:** Random Forest achieved the lowest cross-validation RMSE among the three evaluated models and the strongest validation performance in this experiment.

*Note: Results are specific to the dataset, preprocessing configuration, random seed, and validation split used during the experiment. They are not guarantees of performance on other datasets.*

---

## 🖥️ Application Features

- Interactive Streamlit dashboard
- Dataset upload and exploration
- Automatic feature detection
- Configurable preprocessing
- Multiple regression model comparison
- Hyperparameter tuning
- Model evaluation and visualization
- SHAP explainability
- Feature importance analysis
- Individual predictions
- Batch prediction support
- Dataset-specific model artifact storage

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Application Framework | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Explainability | SHAP |
| Data Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Dataset Support | CSV, Excel |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
House-Price-Prediction-Model/
│
├── app.py
│
├── src/
│   ├── explainability.py
│   ├── generic_pipeline.py
│   ├── model_training.py
│   ├── model_tuning.py
│   ├── prediction.py
│   └── preprocessing.py
│
├── dataset/
│   ├── uci_real_estate/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── models/
│   ├── active_model.json
│   └── datasets/
│
├── notebooks/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
├── README.md
└── EXPERIMENT_COMPARISON.md
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Visual Studio Code (recommended)

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

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

## 🔍 Future Enhancements

- Integration of additional regression algorithms.
- Advanced feature engineering.
- Automated model selection.
- Improved prediction monitoring and reporting.
- Model deployment through cloud platforms.
- Expanded dataset support and validation.

## 👨‍💻 Author

**K A Kaushik**

B.Tech – Artificial Intelligence and Data Science

Interested in Data Science, Machine Learning, Predictive Analytics, and building practical AI-driven applications.

- GitHub: [@Ka1003ik](https://github.com/Ka1003ik)

---

## 📜 License

This project is developed for educational and portfolio purposes.

---

**If you find this project useful, consider giving the repository a ⭐ on GitHub.**
