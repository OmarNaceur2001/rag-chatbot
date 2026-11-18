"""
Shared settings and helpers.

Used by ingest.py (Phase 1) and by rag.py (Phase 2), so both always agree on the same
paths, the same embedding model and the same database.
"""
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ---------- Paths (built from this file's location: they work from any working directory) ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # the project root (rag-chatbot/)
DOCS_DIR = BASE_DIR / "data" / "docs"                # the documents to read
CHROMA_DIR = BASE_DIR / "chroma_db"                  # where the vector database is saved

# ---------- Settings ----------
CHUNK_WORDS = 200                            # target size of one chunk, in words
CHUNK_OVERLAP_WORDS = 40                     # words repeated between two neighbouring chunks

# Read Gemini configuration from the .env file. .env is intentionally never committed.
load_dotenv(BASE_DIR / ".env", override=True, encoding="utf-8-sig")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "docs_gemini").strip() or "docs_gemini"
LEGACY_COLLECTION_NAME = "docs"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()


def get_chroma_client():
    # Telemetry off: nothing leaves the machine (clients like that).
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def get_collection():
    """Open the collection created by ingest.py.

    Always search it with query_embeddings (vectors made by embed_texts), never with query_texts:
    ChromaDB does not remember which embedding model we used, so query_texts would silently
    fall back to its own default model and return wrong results.
    """
    return get_chroma_client().get_collection(COLLECTION_NAME)


def embed_texts(
    texts: list[str],
    provider=None,
    batch_size: int = 100,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed texts with the configured Gemini model and retrieval task type."""
    if provider is None:
        from app.gemini_provider import get_gemini_provider
        provider = get_gemini_provider()
    if task_type == "RETRIEVAL_QUERY":
        return [provider.embed_query(text) for text in texts]
    return provider.embed_documents(texts, batch_size=batch_size)


def has_gemini_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())