"""
Home Practice 2: BERT FAQ Semantic Search.

This script builds a small FAQ semantic search engine using Sentence Transformers.
It returns the top-k most relevant answers from a small knowledge base and uses
a similarity threshold to avoid returning unrelated answers.

Run:
    python 6_bert_faq_semantic_search.py
"""

from sentence_transformers import SentenceTransformer, util


KNOWLEDGE_BASE = [
    "The library is open from 8 AM to 10 PM on weekdays.",
    "Students can borrow up to five books at a time.",
    "Books can be renewed online through the student portal.",
    "Late returns may result in a small daily fine.",
    "Study rooms can be reserved up to one week in advance.",
    "The library provides access to academic journals and research databases.",
    "Printing and scanning services are available on the first floor.",
    "Laptops can be borrowed from the front desk for up to three hours.",
    "Food is not allowed inside the main reading rooms.",
    "Library staff can help students find books and research materials.",
]


def get_top_k_answers(
    query: str,
    kb: list[str],
    kb_embeddings,
    model: SentenceTransformer,
    k: int = 3,
    threshold: float = 0.2,
):
    """Return the top-k FAQ answers for a query, or a fallback message if no answer is relevant."""
    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, kb_embeddings)[0]
    top_results = similarities.topk(k)

    best_score = top_results.values[0].item()
    if best_score < threshold:
        return "I'm sorry, I don't have information on that."

    results = []
    for score, index in zip(top_results.values, top_results.indices):
        results.append(
            {
                "answer": kb[index],
                "score": score.item(),
            }
        )

    return results


def print_answers(query: str, answers) -> None:
    """Print answers returned by get_top_k_answers in a readable format."""
    print("Query:", query)

    if isinstance(answers, str):
        print("Answers:", answers)
        return

    print("Answers:")
    for item in answers:
        print(f"{item['score']:.3f} | {item['answer']}")


def main() -> None:
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding knowledge base...")
    kb_embeddings = model.encode(KNOWLEDGE_BASE, convert_to_tensor=True)

    relevant_query = "How can I reserve a study room?"
    relevant_answers = get_top_k_answers(
        query=relevant_query,
        kb=KNOWLEDGE_BASE,
        kb_embeddings=kb_embeddings,
        model=model,
        k=3,
        threshold=0.2,
    )

    print()
    print_answers(relevant_query, relevant_answers)

    unrelated_query = "How do I fix my car engine?"
    unrelated_answers = get_top_k_answers(
        query=unrelated_query,
        kb=KNOWLEDGE_BASE,
        kb_embeddings=kb_embeddings,
        model=model,
        k=3,
        threshold=0.2,
    )

    print()
    print_answers(unrelated_query, unrelated_answers)


if __name__ == "__main__":
    main()