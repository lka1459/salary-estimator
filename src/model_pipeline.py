import numpy as np
import sys
from typing import List, Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

def select_pipeline(preprocessor) -> Tuple[Pipeline, List[Dict]]:
    while True:
        print("\nSelect a model: \n [1] Support Vector Regression \n [2] Linear Regression \n [3] Exit \n")        
        choice: str = input("Enter choice: ")
        if choice == "1":
            pipeline: Pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVR())
        ])
            param: List[Dict] = [{
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__degree': [1,2,3,4],
            'model__gamma': ['scale', 'auto'],
            'model__C': [0.1, 1, 10, 100, 1000],
            'model__epsilon': [0.01, 0.1, 0.5, 1]
        }]
            return pipeline, param
        elif choice == "2":
            pipeline: Pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
            param: List[Dict] = [{
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, 5, 10, 15, None],
            'model__positive': [True, False],
        }]
            return pipeline, param
        elif choice == "3":
            sys.exit()
        else:
            print("\nInvalid model choice. Please try again.")

    
def fit_and_print(pipe, X_train, X_test, y_train, y_test) -> str:
        pipe.fit(X_train, y_train)
        y_pred: np.array = pipe.predict(X_test)
        mae: str = f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred)}"
        mse: str =  f"Mean Squared Error: ${mean_squared_error(y_test, y_pred)}"
        rmse: str = f"Root Mean Squared Error: ${root_mean_squared_error(y_test, y_pred)}"
        r2: str =  f"R Squared: {r2_score(y_test, y_pred)}"
        return f"{mae} \n{mse} \n{rmse} \n{r2} \n"

def random_search(pipeline, param, X_train, y_train) -> Pipeline:
    clf: RandomizedSearchCV = RandomizedSearchCV(pipeline, param, n_iter=10, cv=5)
    clf.fit(X_train, y_train)
    best_model: Pipeline = clf.best_estimator_
    return best_model

def model_pipeline(preprocessor, X_train, X_test, y_train, y_test) -> Pipeline:
     pipeline, param_grid  = select_pipeline(preprocessor)
     
     print("\n" + "=" * 30)
     print("Baseline model performance:")
     print("=" * 30)

     print(fit_and_print(pipeline, X_train, X_test, y_train, y_test))

     userinput: str = input("Would you like to run hyperparameter tuning? (y/n): ")
     if userinput.lower() == "y":
        rs: Pipeline = random_search(pipeline, param_grid, X_train, y_train)
        return rs
     elif userinput.lower() == "n":
         return pipeline
     else:
         print("Incorrect input.")
         return pipeline
     