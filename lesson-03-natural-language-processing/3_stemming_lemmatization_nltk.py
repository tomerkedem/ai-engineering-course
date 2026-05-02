"""
Simple NLTK demo: stemming vs. lemmatization.

Stemming: rule-based chopping of suffixes to a crude root (may not be a real word).
Lemmatization: dictionary-backed reduction to a canonical lemma, using WordNet.

Run:
    python 3_stemming_lemmatization_nltk.py
"""

# NLTK: natural language toolkit. We use stemmers and the WordNet-based lemmatizer.
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


def ensure_nltk_data() -> None:
    """Download required NLTK resources if missing."""
    # WordNet holds synsets and lemmas for lemmatization; omw-1.4 adds Open Multilingual Wordnet data used by NLTK.
    resources = [
        "wordnet",
        "omw-1.4"
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)


def demo_stemming(words: list[str]) -> None:
    """Show how stemming cuts words to rough roots."""
    # Porter stemmer applies heuristic suffix-stripping rules (fast, no dictionary).
    stemmer = PorterStemmer()
    print("=== Stemming (PorterStemmer) ===")
    for word in words:
        print(f"{word:>12} -> {stemmer.stem(word)}")
    print()


def demo_lemmatization(words: list[str]) -> None:
    """Show lemmatization using POS-aware normalization."""
    # WordNetLemmatizer looks up lemmas; default part-of-speech is noun unless you pass pos=.
    lemmatizer = WordNetLemmatizer()
    print("=== Lemmatization (WordNetLemmatizer) ===")
    for word in words:
        # Without POS hint, nouns are assumed.
        noun_lemma = lemmatizer.lemmatize(word)
        # Verb hint often gives a different, more meaningful base form.
        verb_lemma = lemmatizer.lemmatize(word, pos="v")
        print(f"{word:>12} -> noun:{noun_lemma:<10} verb:{verb_lemma}")
    print()


def main() -> None:
    ensure_nltk_data()

    # Mix of verbs, nouns, adjectives, and irregular forms to contrast stem vs lemma output.
    words = [
        "running",
        "studies",
        "better",
        "wolves",
        "flying",
        "ate",
        "cars",
        "happiest",
    ]

    print("Original words:")
    print(", ".join(words))
    print()

    demo_stemming(words)
    demo_lemmatization(words)

    print("Note:")
    print("- Stemming is faster but can produce non-words (e.g. 'studi').")
    print("- Lemmatization is more linguistic and usually cleaner.")


if __name__ == "__main__":
    main()
