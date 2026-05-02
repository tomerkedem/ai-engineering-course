"""
Named Entity Recognition (NER) with spaCy.

Demonstrates:
1. Built-in NER on a pre-trained English pipeline (en_core_web_sm).
2. A minimal custom NER trained on a few PRODUCT examples.

Setup (once per environment):
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse

import spacy
from spacy.training import Example  # Wraps text + gold entity spans for supervised training
from spacy.util import compounding, minibatch  # Minibatching and growing batch sizes during training


# Text rich in people, orgs, places, dates, etc., so the default English model can show varied entity types.
SAMPLE_TEXT = (
    "Apple Inc. is hiring in Cupertino; Tim Cook met officials in Berlin on 4 April 2025. "
    "Contact: research@apple.com or +1-408-996-1010."
)


def run_pretrained_ner() -> None:
    """Run spaCy's pretrained English NER on SAMPLE_TEXT and print each entity with its label."""
    # Load small English pipeline: tokenizer, tagger, parser, NER, etc.
    nlp = spacy.load("en_core_web_sm")
    # Run the full pipeline; entities appear on doc.ents after the NER component runs.
    doc = nlp(SAMPLE_TEXT)

    print("Built-in NER (en_core_web_sm)\n")
    for ent in doc.ents:
        # Human-readable description of the label (e.g. what "ORG" means).
        explain = spacy.explain(ent.label_) or ""
        print(f"  {ent.text!r:40} | {ent.label_:<10} | {explain}")


def main() -> None:
    """Run the pretrained NER demo, then the custom PRODUCT NER demo."""
    # First: general-purpose entities from the downloaded model.
    run_pretrained_ner()


if __name__ == "__main__":
    main()
