"""Interactive RAG client: retrieve chunks and answer with Claude Haiku 4.5."""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/faiss_index")
TOP_K = 10
LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context chunks.
Rules:
- Answer only using information explicitly stated in the context.
- If the context does not contain enough information to answer, say you cannot answer from the provided context.
- Do not use outside knowledge or make assumptions beyond the context.
- always answer in this format: 1. person - description\n2. person - description\n3. person - description\n..."""


def load_index_and_chunks(index_dir: Path) -> tuple[faiss.Index, list[str], str]:
    """Load FAISS index, chunk texts, and embedding model name from disk."""
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    return index, metadata["chunks"], metadata["model"]


def search(
    question: str,
    index: faiss.Index,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int,
) -> list[tuple[int, float, str]]:
    """Return top_k chunks as (index, L2 distance, text)."""
    query = np.array([model.encode(question)], dtype="float32")
    distances, indices = index.search(query, top_k)
    results = []
    for rank in range(top_k):
        chunk_idx = int(indices[0][rank])
        distance = float(distances[0][rank])
        results.append((chunk_idx, distance, chunks[chunk_idx]))
    return results


def print_results(results: list[tuple[int, float, str]]) -> None:
    for rank, (chunk_idx, distance, text) in enumerate(results, start=1):
        print(f"\n--- Result {rank} (chunk {chunk_idx}, distance {distance:.4f}) ---")
        print(text)


def format_context(results: list[tuple[int, float, str]]) -> str:
    sections = []
    for rank, (chunk_idx, _distance, text) in enumerate(results, start=1):
        sections.append(f"[Chunk {rank} | index {chunk_idx}]\n{text}")
    return "\n\n".join(sections)


def answer_from_context(
    question: str,
    results: list[tuple[int, float, str]],
    client: Anthropic,
) -> str:
    """Ask Claude Haiku 4.5 to answer using only the retrieved chunks."""
    context = format_context(results)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the question using only the context above."
                ),
            }
        ],
    )
    return response.content[0].text


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env or your environment.")

    index, chunks, model_name = load_index_and_chunks(INDEX_DIR)
    model = SentenceTransformer(model_name)
    client = Anthropic(api_key=api_key)

    print("RAG client ready. Ask a question (empty line or 'quit' to exit).\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            break

        results = search(question, index, chunks, model, TOP_K)
        # print_results(results)

        answer = answer_from_context(question, results, client)
        print(f"\n=== Answer ===\n{answer}")

    print("Goodbye.")


if __name__ == "__main__":
    main()
