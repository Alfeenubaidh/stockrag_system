from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.routers.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    ticker: str = Form(...),
    doc_type: str = Form(...),
    filing_date: str = Form(...),
    accession_number: Optional[str] = Form(default=None),
    _auth: None = Depends(require_api_key),
) -> IngestResponse:
    # Lazy imports: ingestion package pulls in pymupdf at import time.
    # Deferring here means a missing/broken PDF dep only fails this endpoint,
    # not server startup — health and query remain available.
    try:
        from ingestion.doc_metadata import DocumentMetadata
        from ingestion.ingest import Ingest
    except ImportError as exc:
        logger.error("Ingestion dependencies unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Ingest unavailable — PDF library not installed: {exc}",
        ) from exc

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    metadata = DocumentMetadata(
        ticker=ticker.upper(),
        doc_type=doc_type,
        filing_date=filing_date,
        accession_number=accession_number or "",
    )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline = Ingest(doc_id=doc_id, metadata=metadata)
        chunks = pipeline.run(tmp_path)
    except ValueError as exc:
        logger.error("Ingest failed doc_id=%r: %s", doc_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected ingest error doc_id=%r: %s", doc_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("Ingest complete doc_id=%r chunks=%d", doc_id, len(chunks))
    return IngestResponse(doc_id=doc_id, chunks_created=len(chunks))
