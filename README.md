# Salary Estimator

A basic machine learning regression project that predicts salary values using structured data.

This project was built as a beginner-friendly regression model to practise the end-to-end machine learning workflow, including data loading, preprocessing, model training, evaluation, and basic hyperparameter tuning.

## Project Overview

The model uses a Salary Prediction dataset from Kaggle. The dataset contains employee salary-related information and is used to practise regression modelling. The target variable is `Salary`, while the remaining columns are used as input features.

## Dataset

The dataset used in this project was taken from Kaggle:

**Salary Prediction dataset** by Rkiattisak  
Source: https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer

The dataset contains employee salary-related information and is used here for a beginner regression modelling project. The target variable is `Salary`.

## Features

- Loads and cleans salary data
- Splits data into training and testing sets
- Automatically detects categorical and numerical columns
- Applies preprocessing using `ColumnTransformer`
- Scales numerical features with `StandardScaler`
- Encodes categorical features with `OneHotEncoder`
- Trains either Linear Regression or SVR
- Evaluates models using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Allows optional hyperparameter tuning using GridSearchCV

## Project Structure

```text
Salary Estimator/
│
├── data/
│   └── salary_data.csv
│
├── src/
│   └── model_pipeline.py
│   └── model_evaluation.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore