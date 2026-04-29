"""
vector_store/qdrant_client.py — Centralised Qdrant client factory.

Single source of truth for:
  - server URL
  - connection check (fail-fast before any operation)
  - startup logging

All modules that need a QdrantClient must call get_qdrant_client().
No local path mode. No dual-mode support. No environment switching.
"""

import logging

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sec_filings"


def get_qdrant_client() -> QdrantClient:
    """
    Return a connected QdrantClient using settings.qdrant_url / settings.qdrant_api_key.

    Raises RuntimeError immediately if the server is unreachable.
    """
    from config.settings import settings

    url = settings.qdrant_url
    api_key = settings.qdrant_api_key or None

    logger.info("Using Qdrant server at %s", url)
    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant unreachable at {url}. "
            "Check QDRANT_URL and QDRANT_API_KEY."
        ) from exc
    return client
