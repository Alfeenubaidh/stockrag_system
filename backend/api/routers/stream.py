from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import get_retrieval_pipeline
from api.routers.auth import require_api_key
from api.routers.rate_limiter import RATE_LIMIT, limiter
from generation.streaming import stream_answer
from retrieval.retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


class StreamQueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


def _sse_generator(
    query: str,
    chunk_dicts: list[dict],
) -> Iterator[str]:
    """Wrap stream_answer tokens as SSE frames, then send [DONE]."""
    try:
        for token in stream_answer(query=query, retrieved_chunks=chunk_dicts):
            # Escape newlines inside token so each SSE event is a single line pair
            safe = token.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
    except Exception as exc:
        logger.error("SSE generation error: %s", exc, exc_info=True)
        yield f"data: [ERROR] {exc}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@router.post("/query/stream")
@limiter.limit(RATE_LIMIT)
def query_stream(
    request: Request,
    req: StreamQueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    _auth: None = Depends(require_api_key),
) -> StreamingResponse:
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

    return StreamingResponse(
        _sse_generator(query=req.question, chunk_dicts=chunk_dicts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx proxy buffering
        },
    )
