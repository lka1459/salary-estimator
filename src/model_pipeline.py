from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

def select_pipeline(preprocessor):
    while True:
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
            'model__C': [0.1, 1, 10, 100, 1000],
            'model__epsilon': [0.01, 0.1, 0.5, 1]
        }]
            return pipeline, param
        elif choice.lower() == "lr":
            pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])
            param = [{
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, 5, 10, 15, None],
            'model__positive': [True, False],
        }]
            return pipeline, param

        else:
            print("Invalid model choice. Please try again.")

    
def fit_and_print(pipe, X_train, X_test, y_train, y_test):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}"
        mse =  f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}"
        r2 =  f"R Squared: {r2_score(y_test, y_pred)}"
        rmse = f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}"
        return mae, mse, r2, rmse

def grid_search(pipeline, param, X_train, y_train):
    clf = GridSearchCV(pipeline, param_grid=param, scoring='r2', cv=5)
    clf.fit(X_train,y_train)
    best_model = clf.best_estimator_    
    return best_model

def random_search(param, X_train, y_train):
    lasso = Lasso()
    random_search = RandomizedSearchCV(lasso, param, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)

def model_pipeline(preprocessor, X_train, X_test, y_train, y_test):
     pipeline, param_grid = select_pipeline(preprocessor)
    
     print("Baseline model performance:")
     print(fit_and_print(pipeline, X_train, X_test, y_train, y_test))

     userinput = input("Would you like to improve the model (y/n): ")
     if userinput.lower() == "y":
        gs = grid_search(pipeline, param_grid, X_train, y_train)
        print("Grid searched model performance:")
        print(fit_and_print(gs, X_train, X_test, y_train, y_test))
        return gs
     elif userinput.lower() == "n":
         return pipeline
     else:
         print("Incorrect input.")
         return pipeline
     