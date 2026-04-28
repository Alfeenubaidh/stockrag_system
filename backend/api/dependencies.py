from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from config.settings import settings
from embeddings.embedder import EmbeddingConfig, EmbeddingPipeline
from retrieval.retrieval import RetrievalConfig, RetrievalPipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    client = QdrantClient(url=settings.qdrant_url)
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant unreachable at {settings.qdrant_url}. "
            "Ensure the qdrant service is running."
        ) from exc
    return client


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingPipeline:
    cfg = EmbeddingConfig(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("Loading embedding model: %s", cfg.model_name)
    return EmbeddingPipeline(config=cfg)


@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    cfg = RetrievalConfig(
        fetch_k=settings.retrieval_fetch_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    return RetrievalPipeline(
        qdrant=get_qdrant(),
        embedder=get_embedder(),
        collection=settings.qdrant_collection,
        config=cfg,
    )
