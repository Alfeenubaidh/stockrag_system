from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class TickerSummary(BaseModel):
    ticker: str
    doc_count: int
    sections: list[str]
    last_updated: str


@router.get("/documents", response_model=list[TickerSummary])
def list_documents() -> list[TickerSummary]:
    """
    Scroll the entire sec_filings collection and return one summary row per ticker:
    chunk count, distinct section names, and the most recent filing_date seen.
    Returns 503 when Qdrant is unreachable so the frontend can show a clear message.
    """
    from qdrant_client import QdrantClient

    try:
        qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        qdrant.get_collections()  # connectivity check
    except Exception as exc:
        logger.warning("documents: Qdrant unreachable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable — is Qdrant running?",
        ) from exc

    collection = settings.qdrant_collection

    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        return []

    # ticker → {doc_count, sections: set, last_updated: str}
    agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"doc_count": 0, "sections": set(), "last_updated": ""}
    )

    offset = None
    try:
        while True:
            points, offset = qdrant.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=["ticker", "section", "filing_date"],
                with_vectors=False,
            )
            for point in points:
                p = point.payload or {}
                ticker = (p.get("ticker") or "unknown").upper()
                row = agg[ticker]
                row["doc_count"] += 1
                section = p.get("section", "")
                if section and section not in ("unknown", ""):
                    row["sections"].add(section)
                filing_date = p.get("filing_date", "")
                if filing_date and filing_date not in ("unknown", "") and filing_date > row["last_updated"]:
                    row["last_updated"] = filing_date

            if offset is None:
                break
    except Exception as exc:
        logger.error("documents scroll failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read collection: {exc}") from exc

    return [
        TickerSummary(
            ticker=ticker,
            doc_count=row["doc_count"],
            sections=sorted(row["sections"]),
            last_updated=row["last_updated"] or "—",
        )
        for ticker, row in sorted(agg.items())
    ]
