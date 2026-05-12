# Telecom Customer Churn ML Pipeline

An end-to-end machine learning project for predicting telecom customer churn and identifying high-risk customer segments using supervised learning, RFM analysis, clustering, feature selection, hyperparameter tuning, and model explainability.

## Project Overview

Customer churn is a major business problem for subscription and telecom companies. This project analyzes customer behavior, usage patterns, spending activity, reload behavior, and inactivity signals to predict whether a customer is likely to churn.

The pipeline includes:

- Exploratory data analysis
- Memory optimization
- Correlation-based feature reduction
- RFM customer segmentation
- K-Means clustering
- Class imbalance handling with SMOTE and random undersampling
- RFECV feature selection
- Model benchmarking across multiple classifiers
- XGBoost hyperparameter tuning with Optuna
- Model evaluation using recall, precision, F1-score, ROC-AUC, and confusion matrix
- SHAP-based model explainability

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- imbalanced-learn
- Matplotlib, Seaborn

## Repository Structure

```bash
Telecom-Customer-Churn-ML-Pipeline/
├── data/
│   └── README.md
├── images/
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── explainability.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py --data data/mobile-churn-data.xlsx --trials 100
```

## Model Approach

### 1. Data Cleaning

The project removes non-predictive identifiers, optimizes memory usage, checks null values, and reduces highly correlated features.

### 2. Customer Segmentation

RFM features are built from reload behavior:

- Recency: reload inactivity days
- Frequency: reload count
- Monetary value: reload amount

K-Means clustering is used to segment customers into behavioral groups.

### 3. Imbalanced Classification

Because churn data is usually imbalanced, the training set is balanced using:

- SMOTE oversampling
- Random undersampling

### 4. Feature Selection

RFECV with Random Forest is used to select the most predictive features based on recall.

### 5. Model Training

The project benchmarks multiple models including:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- LightGBM
- XGBoost

XGBoost is used as the final model after benchmarking and hyperparameter tuning.

### 6. Explainability

SHAP is used to interpret feature influence globally and locally, helping explain why customers are predicted to churn.

## Business Impact

This project helps telecom teams:

- Identify customers at high risk of churn
- Understand churn drivers
- Segment users by customer value
- Prioritize retention campaigns
- Explain model decisions using SHAP

## Key Result

The original notebook concluded that ensemble models, especially XGBoost, performed best for churn prediction, achieving approximately 90% accuracy while improving recall for churners.

## Future Improvements

- Add MLflow experiment tracking
- Deploy model with FastAPI
- Add Streamlit dashboard
- Add automated retraining pipeline
- Add drift detection for production monitoring
