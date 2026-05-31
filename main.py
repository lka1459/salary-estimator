import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.model_pipeline import model_pipeline
from src.model_evaluation import model_evaluation
from src.data_setup import load_and_split_data, build_preprocessor
from src.model_predictor import salary_predictor

def main() -> None:

    print("Salary Estimator")
    print("=" * 30)

    X_train, X_test, y_train, y_test = load_and_split_data()
    preprocessor: ColumnTransformer = build_preprocessor(X_train)

    model: Pipeline = model_pipeline(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
)
    
    while True:
        print("\nWould you like to: \n [1] Predict a Salary \n [2] View model evaluation \n [3] Exit")

        choice: str = input("\nEnter choice: ")
        if choice == "1":
            print(salary_predictor(model))
            return
        elif choice == "2":
            model_evaluation(model, X_test, y_test)
            return
        elif choice == "3":
            sys.exit()
        else:
            print("Invalid choice. Please run the program again.")            

if __name__ == "__main__":
    main()