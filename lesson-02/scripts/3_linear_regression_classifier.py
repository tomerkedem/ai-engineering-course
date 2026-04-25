"""
Linear Regression using scikit-learn for predicting continuous sale prices from living area.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def prepare_data(
    csv_path: str = "data/AmesHousing.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load Ames housing data and return a random train/test split (shuffled, fixed seed).

    Steps:
    1. Read the CSV into a DataFrame.
    2. Keep only the feature (living area) and target (sale price).
    3. Build X as a 2D array (one column per feature) and y as the targets.
    4. Split into train and test so we can fit on train and measure generalization on test.
    """
    df = pd.read_csv(csv_path)
    print(df.head())
    data = df[["Gr Liv Area", "SalePrice"]]
    X = data[["Gr Liv Area"]].values  # shape (n_samples, 1) for sklearn
    y = data["SalePrice"].values
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )  # X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> LinearRegression:
    """
    Fit Ordinary Least Squares linear regression on the training set only.

    Steps:
    1. Create an unfitted LinearRegression (OLS) estimator.
    2. Learn slope and intercept by minimizing squared error on X_train, y_train.
    """
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)  # sets coef_ and intercept_
    return model


def predict_sale_prices(model: LinearRegression, X_train, X_test):
    """
    Apply the fitted model to both splits.

    Steps:
    1. Predict on training data (in-sample / for diagnostics).
    2. Predict on held-out test data (out-of-sample performance).
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train_pred, y_test_pred


def evaluate_quality(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    model: LinearRegression,
):
    """
    Measure how well predictions match true prices on train vs test.

    Steps:
    MSE: average squared difference between y and predictions (lower is better).
    Return the numeric metrics as a dict for reuse or logging.
    """
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
    }
    print(f"\nTraining Results:")
    print(f"  Mean Squared Error: {metrics['train_mse']:.2f}")
    print(f"\nTesting Results:")
    print(f"  Mean Squared Error: {metrics['test_mse']:.2f}")
    return metrics


def show_regression_scatter_and_line_plot(
    model: LinearRegression,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Visualize data and the fitted line: y ≈ slope * area + intercept.

    Steps:
    1. Create a figure and scatter-plot train (blue) and test (green) points.
    2. Build a dense grid of living-area values across the observed min/max.
    3. Predict sale price on that grid and draw the red regression line.
    4. Label axes, title, legend, grid; format y-axis as currency; show the plot.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(X_train, y_train, color="blue", alpha=0.5, label="Training Data", s=20)
    plt.scatter(X_test, y_test, color="green", alpha=0.5, label="Test Data", s=20)
    X_all = np.vstack([X_train, X_test])
    X_line = np.linspace(X_all.min(), X_all.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color="red", linewidth=2, label="Linear Regression Line")
    plt.xlabel("Gr Liv Area (sq ft)", fontsize=12)
    plt.ylabel("Sale Price ($)", fontsize=12)
    plt.title(
        "Linear Regression: House Sale Price vs. Above Ground Living Area",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()
    plt.show()


def linear_regression_example():
    """
    Run the end-to-end demo: data → fit → predict → metrics → one-off example → plot.

    Steps:
    1. Load and split data.
    2. Train the regressor on the training split.
    3. Predict on train and test for evaluation.
    4. Print MSE, R², and coefficients.
    5. Run a single manual prediction for one living-area value.
    6. Show the scatter plot with the fitted line.
    """
    print("=" * 60)
    print("LINEAR REGRESSION EXAMPLE")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data()
    lr_model = train_model(X_train, y_train)
    y_train_pred, y_test_pred = predict_sale_prices(lr_model, X_train, X_test)
    evaluate_quality(
        y_train, y_train_pred, y_test, y_test_pred, lr_model
    )

    # Single-point prediction: one row, one feature (Gr Liv Area)
    sample_input = np.array([[1500]])
    sample_prediction = lr_model.predict(sample_input)
    print(f"\nExample Prediction:")
    print(f"  Input (Gr Liv Area): {sample_input[0][0]} sq ft")
    print(f"  Predicted Sale Price: ${sample_prediction[0]:,.2f}")

    show_regression_scatter_and_line_plot(
        lr_model, X_train, X_test, y_train, y_test
    )
    return lr_model, X_test, y_test, y_test_pred


def main():
    """
    Program entry: print a banner, run the full example pipeline, then print completion.

    Steps:
    1. Announce the script (banner).
    2. Delegate to linear_regression_example() for all ML/plot work.
    3. Print a footer so batch runs clearly reached the end.
    """
    print("\n" + "=" * 60)
    print("SCIKIT-LEARN LINEAR REGRESSION DEMO")
    print("=" * 60)

    linear_regression_example()

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
