import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_FILE = Path("data/starwars_ships_docs.txt")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "starwars_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8")
    return [doc.strip() for doc in text.split("\n\n") if doc.strip()]


def chunk_documents(documents: list[str]) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        chunks.extend(text_splitter.split_text(document))

    return chunks


def build_embeddings(chunks: list[str], model_name: str) -> list[list[float]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()


def save_to_chroma(chunks: list[str], embeddings: list[list[float]]) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_count = collection.count()

    ids = [f"chunk-{existing_count + i}" for i in range(len(chunks))]
    metadatas = [
        {
            "chunk_index": existing_count + i,
            "source": DATA_FILE.name,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Collection had {existing_count} vectors before insert")
    print(f"Added {len(chunks)} new vectors")
    print(f"Collection now has {collection.count()} vectors")



def main() -> None:
    documents = load_documents(DATA_FILE)
    chunks = chunk_documents(documents)
    embeddings = build_embeddings(chunks, EMBEDDING_MODEL)
    save_to_chroma(chunks, embeddings)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved ChromaDB collection to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
