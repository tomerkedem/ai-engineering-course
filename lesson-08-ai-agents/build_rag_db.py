from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Please define it in your .env file.")

print("HF_TOKEN loaded:", bool(hf_token))

CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used by the RAG system.

    The same embedding model must be used when building the vector store
    and when loading it later for search.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def load_and_chunk_documents(data_dir: str = "data"):
    """
    Load all .txt files from the data folder and split them into chunks.
    """
    data_path = Path(__file__).parent / data_dir

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_path}"
        )

    text_files = list(data_path.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"No .txt files found in: {data_path}"
        )

    documents = []

    for file_path in text_files:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks) -> Chroma:
    """
    Build and persist a ChromaDB vector store from document chunks.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load the persisted ChromaDB vector store.

    This function will be used later by the RAG chatbot.
    """
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PERSIST_DIR}. "
            "Run build_rag_db.py first."
        )

    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def main():
    print("Building RAG vector store...")
    print(f"Data folder: {Path(__file__).parent / 'data'}")
    print(f"Persist directory: {CHROMA_PERSIST_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    chunks = load_and_chunk_documents()

    print(f"Loaded and created {len(chunks)} chunks.")

    build_vectorstore(chunks)

    print("Vector store was created successfully.")
    print(f"Saved to: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()