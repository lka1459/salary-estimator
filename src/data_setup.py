from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_and_split_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df: pd.DataFrame = pd.read_csv("data/salary_data.csv")
    df.dropna(inplace=True)

    X: pd.DataFrame = df.drop(columns=['Salary'])
    y: pd.Series = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols: List[str] = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols: List[str] = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor




