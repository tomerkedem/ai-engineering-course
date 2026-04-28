"""
Random Forest Classifier Example using Iris Dataset

This script demonstrates how to:
1. Load and preprocess the Iris dataset
2. Train a Random Forest classifier
3. Evaluate the model performance
4. Visualize individual trees from the forest
5. Analyze feature importance
6. Make predictions on new data
"""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter


def ensure_images_dir(path="images"):
    """Create the output images directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def read_iris(csv_path):
    """Load the Iris CSV into a DataFrame (no preprocessing)."""
    # Resolve CSV path relative to this script's directory for robustness
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_full_path = os.path.join(script_dir, "..", "data", "Iris.csv")
    return pd.read_csv(csv_full_path)


def print_data_overview(df):
    """Print schema, sample rows, shape, and target class balance."""
    print("Loading Iris dataset...")
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["Species"].value_counts())


def make_train_test_split(df, test_size, random_state):
    """Split features vs label, then stratified train/test split; also return feature column names."""
    # Features: all numeric columns except Id; target: Species
    X = df.drop(["Id", "Species"], axis=1)
    y = df["Species"]

    print("\nFeatures:", X.columns.tolist())
    print("\nTarget classes:", y.unique())

    # Stratify keeps class proportions equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, X.columns


def load_data(csv_path="data/Iris.csv", test_size=0.2, random_state=42):
    """Load CSV, print overview, return stratified splits and feature column names."""
    df = read_iris(csv_path)
    print_data_overview(df)
    return make_train_test_split(df, test_size, random_state)


def train(X_train, y_train):
    """Build and fit RandomForestClassifier with regularization-friendly hyperparameters."""
    print("\nTraining Random Forest Classifier...")
    # n_estimators: number of trees; max_features sqrt: random subset of features per split
    # bootstrap + oob_score: out-of-bag estimate of generalization error
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def predict(rf_classifier, X_train, X_test):
    """Predict labels for training and test sets (checks fit vs held-out)."""
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    return y_train_pred, y_test_pred


def feature_importance_df(rf_classifier, X_columns):
    """Return a DataFrame of feature names and importances, sorted high to low."""
    return pd.DataFrame(
        {"Feature": X_columns, "Importance": rf_classifier.feature_importances_}
    ).sort_values("Importance", ascending=False)


def compute_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Compute accuracy on train and test from true labels vs predictions."""
    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
    }


def print_quality_report(rf_classifier, acc, y_test, y_test_pred, feature_importance):
    """Print accuracies, OOB (if available), classification report, confusion matrix, importances."""
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    train_accuracy = acc["train_accuracy"]
    test_accuracy = acc["test_accuracy"]
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # OOB is only meaningful when bootstrap=True (default here)
    if hasattr(rf_classifier, "oob_score_"):
        print(
            f"Out-of-Bag Score: {rf_classifier.oob_score_:.4f} ({rf_classifier.oob_score_ * 100:.2f}%)"
        )
        print("  -> OOB score is an estimate of generalization performance")

    # Per-class precision/recall/F1 on the test set
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Where predictions disagree with true labels (rows=true, cols=predicted)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    print("\nFeature Importance:")
    print(feature_importance)

    return cm


def calculate_quality(
    rf_classifier, X_columns, y_train, y_train_pred, y_test, y_test_pred
):
    """Compute metrics and importances, print report, return a dict of results."""
    acc = compute_accuracy(y_train, y_train_pred, y_test, y_test_pred)
    feature_importance = feature_importance_df(rf_classifier, X_columns)
    cm = print_quality_report(
        rf_classifier, acc, y_test, y_test_pred, feature_importance
    )
    return {
        "train_accuracy": acc["train_accuracy"],
        "test_accuracy": acc["test_accuracy"],
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
    }


def plot_feature_importance(rf_classifier, X_columns):
    """Horizontal bar chart of importances; save under images/."""
    feature_importance = feature_importance_df(rf_classifier, X_columns)

    plt.figure(figsize=(10, 6))
    # Barh: longest bar = most important feature (invert_yaxis shows highest on top)
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    ensure_images_dir()
    plt.savefig(
        "images/random_forest_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "\nFeature importance plot saved as 'images/random_forest_feature_importance.png'"
    )


def plot_sample_trees(rf_classifier, X_columns):
    """Plot up to three estimators from the forest as decision-tree diagrams."""
    print("\nVisualizing sample trees from the forest...")
    n_trees_to_show = min(3, len(rf_classifier.estimators_))
    # Single row of subplots so trees are comparable at a glance
    fig, axes = plt.subplots(1, n_trees_to_show, figsize=(20, 6))
    if n_trees_to_show == 1:
        axes = [axes]

    for i in range(n_trees_to_show):
        # Each estimators_[i] is one DecisionTreeClassifier trained on a bootstrap sample
        plot_tree(
            rf_classifier.estimators_[i],
            feature_names=X_columns,
            class_names=rf_classifier.classes_,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=axes[i],
        )
        axes[i].set_title(
            f"Tree {i + 1} of {len(rf_classifier.estimators_)}",
            fontsize=12,
            fontweight="bold",
        )

    plt.suptitle("Sample Trees from Random Forest", fontsize=16, fontweight="bold")
    plt.tight_layout()
    ensure_images_dir()
    # High DPI for readable node text when zooming
    plt.savefig("images/random_forest_trees.png", dpi=300, bbox_inches="tight")
    print(f"Sample trees visualization saved as 'images/random_forest_trees.png'")


def show_plot(rf_classifier, X_columns):
    """Generate and save both feature-importance and sample-tree figures."""
    plot_feature_importance(rf_classifier, X_columns)
    plot_sample_trees(rf_classifier, X_columns)


def predict_examples(rf_classifier, X_columns):
    """Run the forest on hand-picked rows; show class probs and per-tree vote breakdown."""
    print("\n" + "=" * 50)
    print("PREDICTION EXAMPLES")
    print("=" * 50)

    # Fixed feature vectors in the same order as X_columns (Iris measurements)
    example_samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 3.0, 4.5, 1.5],
        [6.5, 3.0, 5.8, 2.2],
    ]
    example_df = pd.DataFrame(example_samples, columns=X_columns)

    for i, sample in enumerate(example_samples, 1):
        row = example_df.iloc[[i - 1]]  # 2D slice keeps DataFrame API for sklearn
        prediction = rf_classifier.predict(row)
        probabilities = rf_classifier.predict_proba(row)[0]

        print(f"\nExample {i}:")
        print(
            f"  Features: SepalLength={sample[0]}, SepalWidth={sample[1]}, "
            f"PetalLength={sample[2]}, PetalWidth={sample[3]}"
        )
        print(f"  Predicted Species: {prediction[0]}")
        print(f"  Prediction Probabilities:")
        for class_name, prob in zip(rf_classifier.classes_, probabilities):
            print(f"    {class_name}: {prob:.4f} ({prob * 100:.2f}%)")

        # Majority vote: ask each tree for its class, then count votes per species
        X_row = row.to_numpy(dtype=np.float64, copy=False)
        tree_predictions = [
            rf_classifier.classes_[int(np.asarray(tree.predict(X_row))[0])]
            for tree in rf_classifier.estimators_
        ]
        vote_counts = Counter(tree_predictions)
        print(f"  Tree Votes:")
        for class_name in rf_classifier.classes_:
            votes = vote_counts.get(class_name, 0)
            print(
                f"    {class_name}: {votes} trees ({votes / len(rf_classifier.estimators_) * 100:.1f}%)"
            )


def main():
    """End-to-end demo: load data, train, evaluate, plot, explain, and example predictions."""
    # 1. Load Iris, show overview, stratified train/test split
    X_train, X_test, y_train, y_test, X_columns = load_data()
    # 2. Fit random forest on training labels
    rf_classifier = train(X_train, y_train)
    # 3. Class predictions on train and test (for accuracy and reports)
    y_train_pred, y_test_pred = predict(rf_classifier, X_train, X_test)
    # 4. Metrics, confusion matrix, feature table, printed report
    calculate_quality(rf_classifier, X_columns, y_train, y_train_pred, y_test, y_test_pred)
    # 5. Persist importance and sample-tree PNGs
    show_plot(rf_classifier, X_columns)

    print("\n" + "=" * 50)
    print("Script completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
