import pandas as pd
import numpy as np
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

df: pd.DataFrame = pd.read_csv("../data/salary_data.csv")
df.dropna(inplace=True)

X = df.drop(columns=['Salary'])
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

def select_pipeline():
    repeat = True
    while repeat == True:
        choice = input("Would you like to use a SVR model or Linear Regression? (SVR/LR): ")
        if choice.lower() == "svr":
            pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVR())
        ])
            param = [{
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__degree': [1,2,3,4],
            'model__gamma': ['scale', 'auto'],
        }]
            repeat = False
        elif choice.lower() == "lr":
            pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
            param = [{
            'model__fit_intercept': [True, False]
        }]
            repeat = False

        else:
            print("Invalid model choice. Please try again.")
            repeat = True
    return pipeline, param
    
def fit_and_print(pipe):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}"
        mse =  f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}"
        r2 =  f"R Squared: {r2_score(y_test, y_pred)}"
        rmse = f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}"
        return mae, mse, r2, rmse

def grid_search(pipeline, param):
    clf = GridSearchCV(pipeline, param_grid=param, scoring='r2', cv=5)
    clf.fit(X_train,y_train)
    best_model = clf.best_estimator_    
    return best_model

def model_pipeline():
     pipeline, param_grid = select_pipeline()
    
     print("Baseline model performance:")
     print(fit_and_print(pipeline))

     userinput = input("Would you like to improve the model (y/n): ")
     if userinput.lower() == "y":
        gs = grid_search(pipeline, param_grid)
        print("Grid searched model performance:")
        print(fit_and_print(gs))

     elif userinput.lower() == "n":
         return pipeline
     else:
         "Incorrect input."

model_pipeline()

     