"""
Home Practice Bonus: BERT Customer Feedback Clustering.

This script converts customer feedback sentences into sentence embeddings
and uses K-Means clustering to group similar feedback items together.

Run:
    python 7_bert_customer_feedback_clustering.py
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


FEEDBACK_SENTENCES = [
    "I was charged twice for my monthly subscription.",
    "The invoice amount is higher than expected.",
    "I need help understanding my last bill.",
    "My payment failed even though my card is valid.",
    "I was charged for a service I did not use.",
    "The app crashes every time I open the dashboard.",
    "I cannot log in after the latest update.",
    "The search button does not work on my phone.",
    "The screen freezes when I try to upload a file.",
    "The app shows an error when I reset my password.",
    "The new design is very clean and easy to use.",
    "Customer support answered my question quickly.",
    "The app is much faster after the update.",
    "I really like the new notification settings.",
    "The checkout process is simple and clear.",
]


def group_sentences_by_cluster(sentences: list[str], labels) -> dict[int, list[str]]:
    """Group sentences by their assigned cluster label."""
    clusters: dict[int, list[str]] = {}

    for sentence, label in zip(sentences, labels):
        label = int(label)

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(sentence)

    return clusters


def main() -> None:
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding customer feedback...")
    embeddings = model.encode(FEEDBACK_SENTENCES)

    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    clusters = group_sentences_by_cluster(FEEDBACK_SENTENCES, labels)

    for label, sentences in sorted(clusters.items()):
        print(f"\nCluster {label}:")
        for sentence in sentences:
            print("-", sentence)


if __name__ == "__main__":
    main()
