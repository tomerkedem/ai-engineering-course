"""
Home Practice 1: Word2Vec Sentence Embeddings.

This script extends the first Word2Vec notebook by creating a simple
sentence embedding. It removes stop words, averages the vectors of the
remaining words, and compares two sentences using cosine similarity.

Run:
    python 5_word2vec_sentence_embeddings.py
"""

import re

import gensim.downloader as api
import numpy as np


STOP_WORDS = {
    "the", "a", "an", "my", "of", "to", "in", "on",
    "and", "or", "is", "are", "was", "were"
}


def tokenize_and_clean(sentence: str) -> list[str]:
    """Lowercase a sentence, extract English words, and remove stop words."""
    words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
    return [word for word in words if word not in STOP_WORDS]


def sentence_embedding(sentence: str, model) -> np.ndarray | None:
    """Return the average Word2Vec vector for the meaningful words in a sentence."""
    words = tokenize_and_clean(sentence)

    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])

    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def cosine_similarity(vec1: np.ndarray | None, vec2: np.ndarray | None) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0

    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denominator)


def main() -> None:
    print("Loading Word2Vec model...")
    model = api.load("word2vec-google-news-300")

    sentence_a = "The financial institution handled my capital contribution."
    sentence_b = "A primary lender managed the equity funding."

    embedding_a = sentence_embedding(sentence_a, model)
    embedding_b = sentence_embedding(sentence_b, model)

    score = cosine_similarity(embedding_a, embedding_b)

    print()
    print("Sentence A:", sentence_a)
    print("Cleaned A:", tokenize_and_clean(sentence_a))
    print()
    print("Sentence B:", sentence_b)
    print("Cleaned B:", tokenize_and_clean(sentence_b))
    print()
    print("Similarity score:", round(score, 4))


if __name__ == "__main__":
    main()