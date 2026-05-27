import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

def basic_model_evaluation(model, X_test, y_test) -> str:
    y_pred: np.array = model.predict(X_test)
    mae: str = f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}"
    mse: str =  f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}"
    r2: str =  f"R Squared: {r2_score(y_test, y_pred)}"
    rmse: str = f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}"
    return mae, mse, r2, rmse

def plot_residual(model, X_test, y_test) -> None:
    y_pred: np.array = model.predict(X_test)
    residual: pd.Series = y_test - y_pred
    plt.scatter(y_pred, residual)
    plt.axhline(0)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals")
    plt.show()
    return

def model_evaluation(model, X_test, y_test) -> None:
    userInput: str = input("Would you like to run a basic evaluation again? (y/n): ")
    if userInput.lower() == "y":
         print(basic_model_evaluation(model, X_test, y_test))
    elif userInput.lower() == "n":
        pass    
    else:
        print("Invalid input.")
    
    userInput2: str = input("Would you like to run see a visual representation? (y/n): ")
    if userInput2.lower() == "y":
         plot_residual(model, X_test, y_test)
 
    print("Evaluation completed.")
    return
    