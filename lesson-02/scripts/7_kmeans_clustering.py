"""
Minimal K-means example on the Penguins dataset: cluster and plot two features.
"""

import os

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

NUMERICAL_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
N_CLUSTERS = 3
RANDOM_STATE = 42
OUT_PATH = "images/kmeans_clusters.png"


def ensure_output_dir(file_path):
    """Create the parent directory for file_path if it does not exist."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_X(csv_path="data/penguins.csv"):
    df = pd.read_csv(csv_path)
    return df[NUMERICAL_FEATURES].dropna().values


def plot_clusters(X, labels, centroids, out_path=OUT_PATH):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        linewidths=2,
        label="Centroids",
    )
    ax.set_xlabel(NUMERICAL_FEATURES[0])
    ax.set_ylabel(NUMERICAL_FEATURES[1])
    ax.set_title(f"K-Means clustering (K={N_CLUSTERS})")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path} ({len(X)} points)")
    plt.show()


def main():
    X = load_X()
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = model.fit_predict(X)
    plot_clusters(X, labels, model.cluster_centers_)


if __name__ == "__main__":
    main()
