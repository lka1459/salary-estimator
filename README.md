# Salary Estimator

A beginner-friendly machine learning regression project that predicts salary values using structured employee data.

This project was created to practise the end-to-end machine learning workflow, including data loading, preprocessing, model training, evaluation, visualisation, type hints, and basic hyperparameter tuning.

## Dataset

The dataset used in this project was taken from Kaggle:

**Salary Prediction for Beginner**  
Source: https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer

The target variable is `Salary`, while the remaining columns are used as input features.

## Project Overview

The project currently supports two regression models:

- Linear Regression
- Support Vector Regression (SVR)

The user can choose which model to train from the terminal. After training a baseline model, the project can optionally run hyperparameter tuning using `RandomizedSearchCV` to improve model performance.

The project is structured in a modular way, with separate files for data preparation, model training, and model evaluation.

## Features

- Loads and cleans salary data
- Splits data into training and testing sets
- Automatically detects categorical and numerical columns
- Applies preprocessing using `ColumnTransformer`
- Scales numerical features using `StandardScaler`
- Encodes categorical features using `OneHotEncoder`
- Trains Linear Regression or SVR models
- Supports optional hyperparameter tuning using `RandomizedSearchCV`
- Evaluates regression performance using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Includes residual plot visualisation for model error analysis
- Uses Python type hints to improve code readability and maintainability

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