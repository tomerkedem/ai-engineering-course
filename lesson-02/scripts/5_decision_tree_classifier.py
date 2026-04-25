"""
Decision Tree Classifier Example using Iris Dataset

This script demonstrates how to:
1. Load and preprocess the Iris dataset
2. Train a Decision Tree classifier
3. Evaluate the model performance
4. Visualize the decision tree
5. Make predictions on new data
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt




def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load Iris CSV and print a short overview."""
    print("Loading Iris dataset...")
    df = pd.read_csv(csv_path)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df['Species'].value_counts())
    return df


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    """
    Prepare features/target, split, and fit the decision tree (train stage).
    Returns the fitted model and train/test splits plus feature column index.
    """
    X = df.drop(['Id', 'Species'], axis=1)
    y = df['Species']
    print("\nFeatures:", X.columns.tolist())
    print("\nTarget classes:", y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    print("\nTraining Decision Tree Classifier...")
    dt_classifier = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    dt_classifier.fit(X_train, y_train)
    return dt_classifier, X_train, X_test, y_train, y_test, X.columns


def predict(
    dt_classifier: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Run predictions on train and test splits (predict stage)."""
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    return y_train_pred, y_test_pred


def calculate_quality(
    dt_classifier: DecisionTreeClassifier,
    feature_columns: pd.Index,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """Print accuracy, classification report, confusion matrix, feature importance."""
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': dt_classifier.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)


def show_plot(
    dt_classifier: DecisionTreeClassifier,
    feature_names: pd.Index,
    save_path: str = 'images/decision_tree_visualization.png',
) -> None:
    """Render and save the decision tree figure (plot stage)."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_classifier,
        feature_names=feature_names,
        class_names=dt_classifier.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f"\ndone'")


def print_prediction_examples(
    dt_classifier: DecisionTreeClassifier,
    example_samples: list[list[float]],
    feature_columns: pd.Index,
) -> None:
    """Predict and print probabilities for hand-picked feature rows."""
    print("\n" + "="*50)
    print("PREDICTION EXAMPLES")
    print("="*50)

    X_examples = pd.DataFrame(example_samples, columns=feature_columns)
    predictions = dt_classifier.predict(X_examples)
    probas = dt_classifier.predict_proba(X_examples)

    for i, (sample, prediction, probabilities) in enumerate(
        zip(example_samples, predictions, probas), 1
    ):

        print(f"\nExample {i}:")
        print(f"  Features: SepalLength={sample[0]}, SepalWidth={sample[1]}, "
              f"PetalLength={sample[2]}, PetalWidth={sample[3]}")
        print(f"  Predicted Species: {prediction}")
        print(f"  Prediction Probabilities:")
        for class_name, prob in zip(dt_classifier.classes_, probabilities):
            print(f"    {class_name}: {prob:.4f} ({prob*100:.2f}%)")


def main() -> None:
    df = load_dataframe('data/Iris.csv')

    dt_classifier, X_train, X_test, y_train, y_test, feature_columns = train(df)


    y_train_pred, y_test_pred = predict(dt_classifier, X_train, X_test)

    calculate_quality(
        dt_classifier,
        feature_columns,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
    )

    show_plot(dt_classifier, feature_columns)

    example_samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 3.0, 4.5, 1.5],
        [6.5, 3.0, 5.8, 2.2],
    ]
    print_prediction_examples(dt_classifier, example_samples, feature_columns)

    print("\n" + "="*50)
    print("Script completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
