"""Prepare the local Chroma index before the API starts."""

from chromadb.errors import NotFoundError
from app.common import get_collection
from app.gemini_provider import GeminiAuthenticationError
from app.ingest import ingest


def _build_index() -> None:
    try:
        ingest()
    except GeminiAuthenticationError as error:
        raise RuntimeError(
            "Cannot initialize the document index: Gemini rejected GEMINI_API_KEY."
        ) from error


def ensure_index() -> None:
    try:
        collection = get_collection()
    except NotFoundError:
        _build_index()
        return
    if collection.count() == 0:
        _build_index()


if __name__ == "__main__":
    ensure_index()
