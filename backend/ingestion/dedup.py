from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)


def is_duplicate(doc_id: str, qdrant: QdrantClient, collection: str) -> bool:
    """
    Return True if any point with payload.doc_id == doc_id already exists
    in the collection.  Fails open on any Qdrant error (returns False) so
    a transient connectivity blip never silently blocks ingestion.
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if points:
            logger.info("Duplicate detected, skipping doc_id=%s", doc_id)
            return True
        return False
    except Exception as exc:
        logger.warning("Dedup check failed for doc_id=%s — failing open: %s", doc_id, exc)
        return False
