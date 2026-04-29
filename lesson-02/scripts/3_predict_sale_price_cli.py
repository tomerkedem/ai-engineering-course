from pathlib import Path

import joblib
import numpy as np


# ----------------------------------------
# Project paths
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the trained model file
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"


def load_model():
    # ----------------------------------------
    # Load trained model from disk
    # ----------------------------------------
    # The model must already exist.
    # It is created by running:
    # 3_train_linear_regression_model.py
    return joblib.load(MODEL_PATH)


def predict_price(model, living_area):
    # ----------------------------------------
    # Convert user input into model input format
    # ----------------------------------------
    # scikit-learn expects a 2D array:
    # rows = samples
    # columns = features
    input_data = np.array([[living_area]])

    # ----------------------------------------
    # Run prediction
    # ----------------------------------------
    prediction = model.predict(input_data)

    # prediction is an array, so we return the first value
    return prediction[0]


def main():
    # ----------------------------------------
    # 1. Load model once
    # ----------------------------------------
    model = load_model()

    print("Linear Regression model loaded successfully.")
    print("Enter a living area in square feet.")

    # ----------------------------------------
    # 2. Read input from user
    # ----------------------------------------
    user_input = input("Living area: ")

    # ----------------------------------------
    # 3. Convert input to number
    # ----------------------------------------
    living_area = float(user_input)

    # ----------------------------------------
    # 4. Predict sale price
    # ----------------------------------------
    predicted_price = predict_price(model, living_area)

    # ----------------------------------------
    # 5. Print result
    # ----------------------------------------
    print(f"Predicted Sale Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()