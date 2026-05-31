import pandas as pd    
import numpy as np
from typing import Dict
from sklearn.pipeline import Pipeline

def get_int_input(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def get_float_input(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def get_proper_str(prompt: str):
    while True:
        user_input: str = input(prompt).strip()

        if user_input:
            return user_input

        print("Error: Input cannot be empty.")


def salary_predictor(model: Pipeline) -> str:
    print("\n" + "=" * 30)
    print("Salary Prediction:")
    print("=" * 30 + "\n")

    job_title: str = get_proper_str("Enter job title: ")
    age: int = get_int_input("Enter age: ")
    gender: str = get_proper_str("Enter gender: ")    
    years_experience: float = get_float_input("Enter years of experience: ")
    education_level: str = get_proper_str("Enter education level: ")

    data: Dict[str, list[object]] = {
        "Job Title": [job_title],
        "Age": [age],
        "Gender": [gender],
        "Years of Experience": [years_experience],
        "Education Level": [education_level],
    }

    processed_input: pd.DataFrame = pd.DataFrame(data)

    final_prediction: np.ndarray = model.predict(processed_input)

    return f"\nPredicted Salary: ${final_prediction[0]:,.2f}"
