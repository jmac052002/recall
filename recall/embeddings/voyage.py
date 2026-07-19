# recall/embeddings/voyage.py
import voyageai
from functools import lru_cache

from recall.config import settings


@lru_cache(maxsize=1)
def _get_client() -> voyageai.Client:
    """Singleton — one client instance, one connection pool, not one per call."""
    return voyageai.Client(api_key=settings.voyage_api_key)


def embed_text(text: str) -> list[float]:
    """
    Embed a single query string.
    input_type='query' uses Voyage's asymmetric query space —
    optimized for retrieval against stored documents.
    """
    result = _get_client().embed(
        texts=[text],
        model=settings.voyage_model,
        input_type="query",
    )
    return result.embeddings[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Batch embed a list of documents for storage.
    input_type='document' uses Voyage's asymmetric document space.
    Always batch when you have multiple texts — far cheaper on API calls.
    """
    result = _get_client().embed(
        texts=texts,
        model=settings.voyage_model,
        input_type="document",
    )
    return result.embeddings