from src.model_pipeline import model_pipeline
from src.model_evaluation import model_evaluation
from src.data_setup import load_and_split_data, build_preprocessor

def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    preprocessor = build_preprocessor(X_train)

    model = model_pipeline(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
)
    model_evaluation(model, X_test, y_test)


if __name__ == "__main__":
    main()