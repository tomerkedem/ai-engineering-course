"""
Logistic Regression Classifier using scikit-learn
This program demonstrates classification tasks using logistic regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def prepare_data(csv_path: str = "data/Social_Network_Ads.csv"):
    """
    Load the Social Network Ads CSV, extract features and target, print a short
    summary, and split into training and test sets for supervised learning.
    """
    # Load tabular data from disk into a DataFrame.
    df = pd.read_csv(csv_path)
    # Feature matrix: two numeric columns as NumPy arrays (rows = samples).
    X = df[["Age", "EstimatedSalary"]].values
    # Target vector: binary label (0 = did not purchase, 1 = purchased).
    y = df["Purchased"].values

    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: Age, EstimatedSalary")
    print(f"  Target: Purchased (0 = No, 1 = Yes)")
    print(f"  Purchased: {df['Purchased'].sum()} ({df['Purchased'].sum()/len(df)*100:.1f}%)")
    print(
        f"  Not Purchased: {(df['Purchased']==0).sum()} ({(df['Purchased']==0).sum()/len(df)*100:.1f}%)"
    )

    # Hold out 20% for testing; random_state keeps the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Full X, y are returned for plotting the decision boundary on all points.
    return X_train, X_test, y_train, y_test, X, y


def train_classifier(X_train, y_train):
    """
    Build a logistic regression model and fit it so it learns weights that map
    Age and EstimatedSalary to the probability of Purchased=1.
    """
    print("\nTraining Logistic Regression classifier...")
    # max_iter caps optimization steps; random_state fixes any internal RNG use.
    model = LogisticRegression(max_iter=1000, random_state=42)
    # Fit coefficients and intercept by minimizing the logistic loss on training rows.
    model.fit(X_train, y_train)
    return model


def run_predictions(model, X_train, X_test):
    """
    Apply the fitted model to training and test inputs: discrete class labels
    (0 or 1) and, for the test set, class probabilities per row.
    """
    # Predicted class (argmax of learned probabilities) on training data.
    y_train_pred = model.predict(X_train)
    # Same on held-out test data for unbiased evaluation.
    y_test_pred = model.predict(X_test)
    # Per-row probability for each class (column order follows model.classes_).
    y_test_proba = model.predict_proba(X_test)
    return y_train_pred, y_test_pred, y_test_proba


def calculate_quality(model, y_train, y_test, y_train_pred, y_test_pred):
    """
    Evaluate the classifier: accuracy on train vs test, per-class precision/
    recall on test, confusion matrix, learned weights, and a few hand-picked
    examples with predicted class and probabilities.
    """
    # Fraction of correct predictions (training can look optimistic if overfit).
    train_accuracy = accuracy_score(y_train, y_train_pred)
    # Test accuracy reflects generalization to unseen rows.
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Results:")
    print(f"  Accuracy: {train_accuracy:.4f}")

    print(f"\nTesting Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")

    print(f"\nClassification Report:")
    # Precision, recall, F1 per class and support on the test set only.
    print(
        classification_report(
            y_test, y_test_pred, target_names=["Not Purchased", "Purchased"]
        )
    )

    print(f"\nConfusion Matrix:")
    # Rows = true label, columns = predicted label for binary case [[TN, FP], [FN, TP]].
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

    # Fixed (Age, Salary) tuples to illustrate predictions outside the CSV.
    sample_inputs = np.array(
        [
            [25, 50000],
            [45, 120000],
            [30, 30000],
        ]
    )
    sample_predictions = model.predict(sample_inputs)
    sample_proba = model.predict_proba(sample_inputs)

    print(f"\nExample Predictions:")
    for i, (age, salary) in enumerate(sample_inputs):
        pred = sample_predictions[i]
        proba = sample_proba[i]
        print(f"  Age: {age}, Salary: ${salary:,}")
        print(f"    Predicted: {'Purchased' if pred == 1 else 'Not Purchased'}")
        print(f"    Probability (Not Purchased): {proba[0]:.4f}")
        print(f"    Probability (Purchased): {proba[1]:.4f}")

    # Return confusion matrix for the heatmap in show_plot.
    return cm


def show_plot(model, X, y, cm):
    """
    Draw a two-panel figure: (1) feature space with filled decision regions and
    scatter of true labels; (2) confusion matrix as an annotated heatmap.
    """
    plt.figure(figsize=(14, 6))

    # --- Left: decision boundary in (Age, EstimatedSalary) space ---
    plt.subplot(1, 2, 1)
    # Grid step for building a dense set of (age, salary) points to classify.
    h = 0.5
    # Extend ranges slightly so points are not clipped at the plot edge.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )

    # Predict class for every grid cell, then reshape back to 2D for contourf.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Background colors show which side of the boundary the model assigns to each class.
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Overlay actual samples: red circles vs blue squares by true label.
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="red",
        marker="o",
        label="Not Purchased",
        alpha=0.6,
        s=30,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="blue",
        marker="s",
        label="Purchased",
        alpha=0.6,
        s=30,
    )

    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Estimated Salary ($)", fontsize=12)
    plt.title("Logistic Regression: Decision Boundary", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    # Format salary axis with dollar signs and thousands separators.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # --- Right: confusion matrix visualization ---
    plt.subplot(1, 2, 2)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Not Purchased", "Purchased"])
    plt.yticks(tick_marks, ["Not Purchased", "Purchased"])

    # Pick text color so counts stay readable on light vs dark cells.
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14,
            fontweight="bold",
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.show()


def logistic_regression_classifier_example():
    """
    End-to-end demo: load and split data, train logistic regression, predict on
    train/test, print metrics and coefficients, plot boundary and confusion matrix.
    Uses Social Network Ads: predict Purchased from Age and EstimatedSalary.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFIER EXAMPLE")
    print("=" * 60)

    # Load CSV, build X/y, print summary, 80/20 train-test split.
    X_train, X_test, y_train, y_test, X, y = prepare_data()

    # Fit the classifier on training rows only.
    log_reg_model = train_classifier(X_train, y_train)

    # Get hard labels (and test probabilities; discard here if unused downstream).
    y_train_pred, y_test_pred, _y_test_proba = run_predictions(
        log_reg_model, X_train, X_test
    )

    # Print scores, report, matrix, weights, examples; get cm for plotting.
    cm = calculate_quality(
        log_reg_model, y_train, y_test, y_train_pred, y_test_pred
    )

    # Visualize learned separation and test-set confusion counts.
    show_plot(log_reg_model, X, y, cm)

    # Expose model and test outputs for callers that import this module.
    return log_reg_model, X_test, y_test, y_test_pred


def main():
    """
    Script entry point: print a banner, run the full example pipeline, then a
    completion banner. When run as a program (not imported), this is what executes.
    """
    print("\n" + "=" * 60)
    print("SCIKIT-LEARN LOGISTIC REGRESSION CLASSIFIER DEMO")
    print("=" * 60)

    # Run data → train → evaluate → plot; unpack return values for possible extension.
    log_reg_model, X_test, y_test, y_pred = logistic_regression_classifier_example()

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 60)


# Run the demo only when this file is executed directly (`python .../4_logistic_regression_classifier.py`).
if __name__ == "__main__":
    main()
