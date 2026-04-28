"""
qdrant_store.py — Qdrant upsert layer (production-safe)
"""

import logging
import uuid
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    OptimizersConfigDiff,
)

from vector_store.payload_schema import chunk_to_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field schema (correct types)
# ---------------------------------------------------------------------------

FIELD_SCHEMA = {
    "ticker": PayloadSchemaType.KEYWORD,
    "doc_type": PayloadSchemaType.KEYWORD,
    "section": PayloadSchemaType.KEYWORD,
    "filing_timestamp": PayloadSchemaType.INTEGER,
}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "sec_filings",
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
    ):
        self.client = client
        self.collection = collection_name
        self.vector_size = vector_size
        self.distance = distance

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection in existing:
            info = self.client.get_collection(self.collection)
            actual_size = info.config.params.vectors.size

            if actual_size != self.vector_size:
                raise RuntimeError(
                    f"Vector size mismatch: existing={actual_size}, expected={self.vector_size}. "
                    "Delete collection manually if intentional."
                )

            logger.info(f"Collection '{self.collection}' exists")
            return

        # Create collection
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
        )

        # Create indexes with correct schema
        for field, schema in FIELD_SCHEMA.items():
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )

        logger.info(f"Collection '{self.collection}' created")

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        embedded_chunks: List[Dict],
        batch_size: int = 128,
    ) -> None:
        self.ensure_collection()

        total = len(embedded_chunks)

        for batch_start in range(0, total, batch_size):
            batch = embedded_chunks[batch_start : batch_start + batch_size]
            points = []

            for chunk in batch:
                chunk_id = (
                    chunk.get("chunk_id")
                    or f"{chunk['doc_id']}_{chunk['chunk_index']}"
                )

                embedding = chunk.get("embedding")

                if embedding is None:
                    raise ValueError(f"Missing embedding | {chunk_id}")

                if not isinstance(embedding, (list, tuple)):
                    raise TypeError(f"Invalid embedding type | {chunk_id}")

                if len(embedding) != self.vector_size:
                    raise ValueError(
                        f"Embedding size mismatch: {len(embedding)} != {self.vector_size} | {chunk_id}"
                    )

                if any(not isinstance(x, (float, int)) for x in embedding):
                    raise TypeError(f"Non-numeric embedding values | {chunk_id}")

                payload = chunk_to_payload(chunk)
                payload["chunk_id"] = chunk_id

                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=embedding,
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True,
                )
            except Exception as e:
                logger.error(f"Upsert failed at batch {batch_start}: {e}")
                raise

            logger.info(
                f"Upserted {batch_start}-{batch_start + len(batch) - 1} / {total}"
            )

        logger.info(f"Upsert complete: {total} points")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }