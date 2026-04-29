from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


# ----------------------------------------
# Project paths
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "AmesHousing.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "linear_regression_model.pkl"

def main():
    # ----------------------------------------
    # 1. Load dataset
    # ----------------------------------------
    df = pd.read_csv(DATA_PATH)

    # ----------------------------------------
    # 2. Select relevant columns
    # ----------------------------------------
    data = df[["Gr Liv Area", "SalePrice"]].dropna()

    X = data[["Gr Liv Area"]].values
    y = data["SalePrice"].values

    # ----------------------------------------
    # 3. Train model
    # ----------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    # ----------------------------------------
    # 4. Create models directory if needed
    # ----------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 5. Save model to file
    # ----------------------------------------
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved successfully.")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
