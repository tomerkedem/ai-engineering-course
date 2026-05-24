import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "starwars_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context chunks.
Rules:
- Answer only using information explicitly stated in the context.
- If the context does not contain enough information to answer, say you cannot answer from the provided context.
- Do not use outside knowledge or make assumptions beyond the context.
- Answer in a clear structure that fits the user's question.
"""


def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def search(question: str, collection, model: SentenceTransformer, top_k: int) -> list[dict]:
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append(
            {
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    return retrieved


def format_context(results: list[dict]) -> str:
    sections = []
    for rank, item in enumerate(results, start=1):
        sections.append(
            f"[Chunk {rank} | metadata: {item['metadata']}]\n{item['document']}"
        )
    return "\n\n".join(sections)


def answer_from_context(question: str, results: list[dict], client: Anthropic) -> str:
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

def print_results(results: list[dict]) -> None:
    for rank, item in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Distance: {item['distance']}")
        print(f"Metadata: {item['metadata']}")
        print(item["document"])

 
def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env or your environment.")

    collection = load_collection()
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = Anthropic(api_key=api_key)

    print("ChromaDB RAG client ready. Ask a question (empty line or 'quit' to exit).\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            break

        results = search(question, collection, model, TOP_K)
        answer = answer_from_context(question, results, client)

        print(f"\n=== Answer ===\n{answer}")
        print_results(results)

    print("Goodbye.")


if __name__ == "__main__":
    main()
