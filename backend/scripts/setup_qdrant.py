"""
scripts/setup_qdrant.py — One-time (idempotent) Qdrant collection setup.

Usage:
    python scripts/setup_qdrant.py
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_qdrant")


def main() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PayloadSchemaType

    from config.settings import settings
    from vector_store.qdrant_store import QdrantStore

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    try:
        client.get_collections()
    except Exception as exc:
        logger.error("Qdrant unreachable at %s: %s", settings.qdrant_url, exc)
        sys.exit(1)

    store = QdrantStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=384,
    )
    store.ensure_collection()

    # Idempotent — safe to call on an existing collection.
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="ticker",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Payload index ensured: ticker (keyword)")

    print(f"Collection ready: {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
