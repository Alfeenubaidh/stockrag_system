from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.dependencies import get_retrieval_pipeline
from api.routers.auth import require_api_key
from api.routers.rate_limiter import RATE_LIMIT, limiter
from generation.generator import generate_answer
from retrieval.retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: int


def _extract_citations(gen_output: dict[str, Any]) -> list[str]:
    """Pull citation tags from answers text; fall back to answer keys."""
    import re

    citations: list[str] = []
    for ticker, text in gen_output.get("answers", {}).items():
        if not isinstance(text, str):
            continue
        found = re.findall(r"\[[^\]]+\]", text)
        citations.extend(found)
    return list(dict.fromkeys(citations))  # deduplicated, order-preserving


_DISCLAIMER = "This is not financial advice. For informational purposes only."


def _flatten_answer(gen_output: dict[str, Any]) -> str:
    answers = gen_output.get("answers", {})
    comparison = gen_output.get("comparison", "")

    parts: list[str] = []
    for ticker, text in answers.items():
        if isinstance(text, str) and text.strip():
            parts.append(f"**{ticker}**: {text.strip()}")

    if comparison and comparison not in ("Not found in filings", ""):
        parts.append(f"\n**Comparison**: {comparison}")

    body = "\n\n".join(parts) if parts else "No information found in retrieved filings."
    disclaimer = gen_output.get("disclaimer", _DISCLAIMER)
    return f"{body}\n\n---\n{disclaimer}"


@router.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
def query(
    request: Request,
    req: QueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    _auth: None = Depends(require_api_key),
) -> QueryResponse:
    t0 = time.monotonic()

    tickers = [req.ticker.upper()] if req.ticker else None

    try:
        chunks = pipeline.retrieve(query=req.question, tickers=tickers, top_k=req.top_k)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    chunk_dicts = [
        {
            "ticker": r.metadata.get("ticker", ""),
            "section": r.section,
            "text": r.text,
            "score": r.score,
            "doc_type": r.doc_type,
            "filing_date": r.metadata.get("filing_date", ""),
        }
        for r in chunks
    ]

    try:
        gen_output = generate_answer(query=req.question, retrieved_chunks=chunk_dicts)
    except RuntimeError as exc:
        logger.error("Generator failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected generator error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)

    return QueryResponse(
        answer=_flatten_answer(gen_output),
        citations=_extract_citations(gen_output),
        latency_ms=latency_ms,
    )
