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


def parse_living_area(user_input: str) -> float:
    # ----------------------------------------
    # Convert CLI input into a valid number
    # ----------------------------------------
    # input() always returns text.
    # We need to convert it to float before using it.
    value = float(user_input)

    # ----------------------------------------
    # Validate business meaning
    # ----------------------------------------
    # A house living area must be greater than zero.
    if value <= 0:
        raise ValueError("Living area must be greater than zero.")

    return value


def predict_price(model, living_area: float) -> float:
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
    print("Type 'exit' to quit.")

    # ----------------------------------------
    # 2. Keep asking the user for input
    # ----------------------------------------
    while True:
        user_input = input("\nLiving area: ").strip()

        # ----------------------------------------
        # 3. Allow the user to exit the program
        # ----------------------------------------
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            # ----------------------------------------
            # 4. Parse and validate input
            # ----------------------------------------
            living_area = parse_living_area(user_input)

            # ----------------------------------------
            # 5. Predict sale price
            # ----------------------------------------
            predicted_price = predict_price(model, living_area)

            # ----------------------------------------
            # 6. Print result
            # ----------------------------------------
            print(f"Predicted Sale Price: ${predicted_price:,.2f}")

        except ValueError:
            # ----------------------------------------
            # Handle illegal input without crashing
            # ----------------------------------------
            print("Invalid input. Please enter a positive number, for example: 1500")


if __name__ == "__main__":
    main()

