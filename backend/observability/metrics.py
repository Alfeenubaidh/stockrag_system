from __future__ import annotations

import logging

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom metrics — import these from other modules to record observations
# ---------------------------------------------------------------------------

retrieval_chunk_count = Histogram(
    "stockrag_retrieval_chunk_count",
    "Number of chunks returned per retrieval call",
    buckets=[1, 3, 5, 8, 10, 15, 20, 50],
)

generation_latency_seconds = Histogram(
    "stockrag_generation_latency_seconds",
    "Wall-clock seconds spent in LLM generation",
    buckets=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

generation_requests_total = Counter(
    "stockrag_generation_requests_total",
    "Total generation requests",
    ["status"],  # labels: ok | error
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_metrics(app: FastAPI) -> None:
    """
    Attach Prometheus instrumentation to the FastAPI app.
    Exposes /metrics endpoint for Prometheus scraping.
    """
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    logger.info("Prometheus metrics enabled at /metrics")
