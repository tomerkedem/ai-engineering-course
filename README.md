# AI Engineering Course

This repository contains the practical labs from the AI Engineering course.

The goal of this repository is to complement the theoretical material
by providing runnable examples that demonstrate how Machine Learning
concepts are applied in practice.

---

## Structure

Currently, the repository includes:

- Lesson 02 – Machine Learning Lab

Each lab focuses on a specific topic covered in the course
and is designed to be explored by running and modifying the code.

---

## Learning Approach

The labs are intended to be used in the following way:

Read → Run → Modify → Observe

Do not just read the code.
Run it, change it, and explore how the results are affected.

---

## Requirements

Each lab includes its own setup instructions.
In most cases, you will need to install dependencies using:

```bash
pip install -r requirements.txt
```

Notes

This repository will be updated as the course progresses
and additional labs are introduced.

## Home Practice 1: Word2Vec Sentence Embeddings

This script extends the first Word2Vec notebook by creating a simple sentence embedding.
It removes stop words, averages the vectors of the remaining words, and compares two sentences using cosine similarity.

Run:

```bash
python 5_word2vec_sentence_embeddings.py
```

## Home Practice 2: BERT FAQ Semantic Search

This script builds a small FAQ semantic search engine using Sentence Transformers.
It returns the top-k most relevant answers from a small knowledge base and uses
a similarity threshold to avoid returning unrelated answers.

Run:

```bash
python 6_bert_faq_semantic_search.py
