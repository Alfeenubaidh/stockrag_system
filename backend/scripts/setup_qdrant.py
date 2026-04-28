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
    from config.settings import settings
    from vector_store.qdrant_client import get_qdrant_client
    from vector_store.qdrant_store import QdrantStore

    try:
        client = get_qdrant_client()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    store = QdrantStore(
        client=client,
        collection_name=settings.qdrant_collection,
        vector_size=384,
    )
    store.ensure_collection()
    print(f"Collection ready: {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
