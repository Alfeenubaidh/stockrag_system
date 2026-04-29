from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routers import documents, health, ingest, query, stream
from api.routers.rate_limiter import limiter
from config.settings import settings
from observability.metrics import setup_metrics

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Start nightly EDGAR scheduler if watchlist is configured
    scheduler = None
    if getattr(settings, "watchlist_tickers", []):
        try:
            from ingestion.scheduler import start as start_scheduler
            scheduler = start_scheduler()
        except Exception as exc:
            logger.warning("Scheduler failed to start: %s", exc)
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="StockRAG API",
    description="Financial RAG system — retrieval and generation over SEC filings.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://stockrag-system.vercel.app",
    ],
    # Covers Vercel preview deployments (e.g. stockrag-system-abc123-user.vercel.app)
    allow_origin_regex=r"https://stockrag-system-[a-z0-9\-]+\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(stream.router)
app.include_router(ingest.router)
app.include_router(documents.router)

# Prometheus — must be called after routers are registered
setup_metrics(app)
