# Salary Estimator

A beginner-friendly machine learning regression project that predicts salary values using structured employee data.

This project was created to practise the end-to-end machine learning workflow, including data loading, preprocessing, model training, evaluation, visualisation, and basic hyperparameter tuning.

## Dataset

The dataset used in this project was taken from Kaggle:

**Salary Prediction for Beginner**  
Source: https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer

The target variable is `Salary`, while the remaining columns are used as input features.

## Project Overview

The project currently supports two regression models:

- Linear Regression
- Support Vector Regression (SVR)

The user can choose which model to train. After training a baseline version, the project can optionally run GridSearchCV to tune the selected model.

## Features

- Loads and cleans salary data
- Splits data into training and testing sets
- Automatically detects categorical and numerical columns
- Applies preprocessing using `ColumnTransformer`
- Scales numerical features using `StandardScaler`
- Encodes categorical features using `OneHotEncoder`
- Trains Linear Regression or SVR models
- Evaluates regression performance using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Supports optional GridSearchCV tuning
- Includes residual plot visualisation for model error analysis

## Project Structure

```text
Salary Estimator/
│
├── data/
│   └── salary_data.csv
│
├── src/
│   ├── data_setup.py
│   ├── model_pipeline.py
│   └── model_evaluation.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore