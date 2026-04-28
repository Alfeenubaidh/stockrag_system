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

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sec_filings"


def get_qdrant_client() -> QdrantClient:
    """
    Return a connected QdrantClient pointed at the Docker server.

    Raises RuntimeError immediately if the server is unreachable —
    prevents silent failures deep inside the pipeline.
    """
    logger.info("Using Qdrant server at %s", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant server not running on {QDRANT_URL}. "
            "Start it with: docker run -p 6333:6333 qdrant/qdrant"
        ) from exc
    return client
