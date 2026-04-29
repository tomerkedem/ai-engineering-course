from pathlib import Path

import joblib
import numpy as np


# ----------------------------------------
# Project paths
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def parse_living_area(user_input: str) -> float:
    value = float(user_input)

    if value <= 0:
        raise ValueError("Living area must be greater than zero.")

    return value


def predict_price(model, living_area: float) -> float:
    input_data = np.array([[living_area]])
    prediction = model.predict(input_data)
    return prediction[0]


def main():
    # ----------------------------------------
    # Load model once (startup time)
    # ----------------------------------------
    model = load_model()

    print("Model loaded successfully.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nLiving area: ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            living_area = parse_living_area(user_input)
            predicted_price = predict_price(model, living_area)

            print(f"Predicted Sale Price: ${predicted_price:,.2f}")

        except ValueError:
            print("Invalid input. Please enter a positive number.")


if __name__ == "__main__":
    main()
