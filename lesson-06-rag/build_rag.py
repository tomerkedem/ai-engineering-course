"""Build a FAISS index from Star Wars ship documents."""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FILE = Path("data/starwars_ships_docs.txt")
INDEX_DIR = Path("data/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_WORDS = 300
OVERLAP_WORDS = 50


def load_documents(file_path: Path) -> list[str]:
    """Load documents separated by blank lines."""
    text = file_path.read_text(encoding="utf-8")
    documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
    return documents


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking them apart later."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def count_words(text: str) -> int:
    return len(text.split())


def chunk_document(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Chunk one document into overlapping sentence-based pieces."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        chunk_sentences = []
        word_count = 0
        end_idx = start_idx

        while end_idx < len(sentences):
            sentence = sentences[end_idx]
            sentence_words = count_words(sentence)

            if chunk_sentences and word_count + sentence_words > max_words:
                break

            chunk_sentences.append(sentence)
            word_count += sentence_words
            end_idx += 1

        chunks.append(" ".join(chunk_sentences))

        if end_idx >= len(sentences):
            break

        overlap_count = 0
        next_start = end_idx
        for idx in range(end_idx - 1, start_idx - 1, -1):
            overlap_count += count_words(sentences[idx])
            next_start = idx
            if overlap_count >= overlap_words:
                break

        if next_start <= start_idx:
            next_start = start_idx + 1

        start_idx = next_start

    return chunks


def chunk_documents(documents: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Chunk all documents and return a flat list of text chunks."""
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document, max_words, overlap_words))
    return all_chunks


def build_embeddings(chunks: list[str], model_name: str) -> np.ndarray:
    """Embed text chunks with the given sentence-transformers model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")


def save_faiss_index(embeddings: np.ndarray, chunks: list[str], output_dir: Path) -> None:
    """Save the FAISS index and chunk metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = {"model": EMBEDDING_MODEL, "chunks": chunks}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    documents = load_documents(DATA_FILE)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        chunks.extend(text_splitter.split_text(document))

    embeddings = build_embeddings(chunks, EMBEDDING_MODEL)
    save_faiss_index(embeddings, chunks, INDEX_DIR)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
