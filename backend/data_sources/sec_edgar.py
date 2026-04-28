"""
data_sources/sec_edgar.py — Fetch and ingest SEC EDGAR filings.

No API key required. EDGAR is a public US government service.
Rate limit: 10 requests/second — enforced via _SEC_SLEEP.
User-Agent is mandatory per EDGAR policy.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = "StockRAG research@stockrag.com"
_SEC_SLEEP = 0.12  # 10 req/s cap; 0.12 s keeps us at ~8 req/s
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_cik_cache: dict[str, str] = {}  # ticker.upper() → zero-padded 10-digit CIK


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> requests.Response:
    time.sleep(_SEC_SLEEP)
    return requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=20,
        **kwargs,
    )


def _resolve_cik(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for *ticker*, or None if not found."""
    key = ticker.upper()
    if key in _cik_cache:
        return _cik_cache[key]

    resp = _get(_TICKERS_URL)
    resp.raise_for_status()
    for item in resp.json().values():
        if item.get("ticker", "").upper() == key:
            cik = str(item["cik_str"]).zfill(10)
            _cik_cache[key] = cik
            return cik

    logger.warning("sec_edgar: CIK not found for ticker %s", ticker)
    return None



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_latest_filing(ticker: str, doc_type: str = "10-K") -> Optional[dict]:
    """
    Return metadata for the most recent *doc_type* filing for *ticker*.

    Return value::

        {
            "ticker":               "AAPL",
            "doc_type":             "10-K",
            "filing_date":          "2024-09-28",
            "accession_number":     "0000320193-24-000123",
            "primary_document_url": "https://www.sec.gov/Archives/...",
        }

    Returns None if the ticker is unknown or no matching filing exists.
    """
    cik = _resolve_cik(ticker)
    if cik is None:
        return None

    resp = _get(_SUBMISSIONS_URL.format(cik=cik))
    resp.raise_for_status()
    recent = resp.json().get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    target = doc_type.upper()

    for form, filing_date, accession, primary_doc in zip(
        forms, dates, accessions, primary_docs
    ):
        if form.upper() != target:
            continue

        # Build the primary document URL
        acc_nodashes = accession.replace("-", "")
        cik_int = int(cik)
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data"
            f"/{cik_int}/{acc_nodashes}/{primary_doc}"
        )

        return {
            "ticker": ticker.upper(),
            "doc_type": doc_type,
            "filing_date": filing_date,
            "accession_number": accession,
            "primary_document_url": doc_url,
        }

    logger.info("sec_edgar: no %s filing found for %s", doc_type, ticker)
    return None


def download_filing(
    filing: dict,
    output_dir: str = "data/raw/pdfs",
) -> Optional[Path]:
    """
    Download the primary document from *filing* and save it to *output_dir*.

    Filename convention: ``{TICKER}_{YEAR}_{DOCTYPE}.{ext}``
    Extension is taken directly from the primary_document_url (.pdf or .htm).
    e.g. ``AAPL_2024_10K.htm``  or  ``AAPL_2024_10K.pdf``

    Returns the Path to the saved file, or None on failure.
    Already-existing files are skipped (dedup by filename).
    """
    ticker = filing["ticker"].upper()
    doc_type = filing["doc_type"].replace("-", "")   # "10-K" → "10K"
    year = filing["filing_date"][:4]
    url = filing["primary_document_url"]

    suffix = Path(url.split("?")[0]).suffix.lower() or ".htm"
    if suffix not in (".pdf", ".htm", ".html"):
        suffix = ".htm"

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{ticker}_{year}_{doc_type}{suffix}"
    dest = dest_dir / filename

    if dest.exists():
        logger.info("sec_edgar: file already exists, skipping — %s", dest)
        return dest

    logger.info("sec_edgar: downloading %s → %s", url, dest)
    try:
        resp = _get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("sec_edgar: saved %s (%d bytes)", dest, dest.stat().st_size)
        return dest
    except Exception as exc:
        logger.error("sec_edgar: download failed for %s: %s", url, exc)
        dest.unlink(missing_ok=True)
        return None


def fetch_and_ingest(ticker: str, doc_type: str = "10-K") -> bool:
    """
    End-to-end: resolve → download → ingest into Qdrant.

    Handles both PDF and HTM primary documents:
    - .pdf  → PDFParser via Ingest pipeline
    - .htm  → parse_html_filing → ChunkValidator → embed → upsert

    Returns True if a new filing was ingested, False if skipped or failed.
    """
    logger.info("sec_edgar.fetch_and_ingest: ticker=%s doc_type=%s", ticker, doc_type)

    filing = get_latest_filing(ticker, doc_type)
    if filing is None:
        logger.warning("sec_edgar: no %s filing found for %s", doc_type, ticker)
        return False

    logger.info(
        "sec_edgar: found %s filed %s (accession %s)",
        doc_type, filing["filing_date"], filing["accession_number"],
    )

    path = download_filing(filing)
    if path is None:
        logger.error("sec_edgar: download failed for %s %s", ticker, doc_type)
        return False

    try:
        from config.settings import settings
        from embeddings.embedder import EmbeddingConfig, EmbeddingPipeline
        from ingestion.dedup import is_duplicate
        from vector_store.qdrant_client import get_qdrant_client
        from vector_store.qdrant_store import QdrantStore

        year = filing["filing_date"][:4]
        doc_type_slug = doc_type.replace("-", "")
        doc_id = f"{ticker.upper()}_{year}_{doc_type_slug}"

        qdrant = get_qdrant_client()
        if is_duplicate(doc_id, qdrant, settings.qdrant_collection):
            logger.info("sec_edgar: already ingested doc_id=%s — skipping (ok)", doc_id)
            return True

        ext = path.suffix.lower()

        if ext == ".pdf":
            chunks = _ingest_pdf(path, doc_id, ticker, doc_type, filing)
        elif ext in (".htm", ".html"):
            chunks = _ingest_htm(path, doc_id, ticker, doc_type, filing)
        else:
            logger.error("sec_edgar: unsupported file type %s for %s", ext, doc_id)
            return False

        if not chunks:
            logger.error("sec_edgar: no chunks produced for %s", doc_id)
            return False

        chunk_dicts = [
            {
                "chunk_id":         f"{c.doc_id}_{c.chunk_index}",
                "doc_id":           c.doc_id,
                "ticker":           c.ticker or ticker.upper(),
                "doc_type":         c.doc_type or doc_type,
                "filing_date":      c.filing_date or filing["filing_date"],
                "accession_number": c.accession_number or filing["accession_number"],
                "section":          c.section or "unknown",
                "start_page":       c.start_page,
                "end_page":         c.end_page,
                "chunk_index":      c.chunk_index,
                "text":             c.text,
            }
            for c in chunks
        ]

        embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=128))
        embedded = embedder.embed_chunks(chunk_dicts)

        store = QdrantStore(client=qdrant, collection_name=settings.qdrant_collection)
        store.upsert_chunks(embedded)

        logger.info("sec_edgar: ingested %d chunks for %s", len(chunks), doc_id)
        return True

    except Exception as exc:
        logger.error("sec_edgar: ingest pipeline failed for %s: %s", ticker, exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Format-specific ingest helpers
# ---------------------------------------------------------------------------

def _ingest_pdf(
    path: Path,
    doc_id: str,
    ticker: str,
    doc_type: str,
    filing: dict,
):
    """Run the standard PDF → Ingest pipeline. Returns List[Chunk]."""
    from ingestion.doc_metadata import DocumentMetadata
    from ingestion.ingest import Ingest

    metadata = DocumentMetadata(
        ticker=ticker.upper(),
        doc_type=doc_type,
        filing_date=filing["filing_date"],
        accession_number=filing["accession_number"],
    )
    logger.info("sec_edgar: PDF ingest pipeline for %s", doc_id)
    return Ingest(doc_id=doc_id, metadata=metadata).run(str(path))


def _ingest_htm(
    path: Path,
    doc_id: str,
    ticker: str,
    doc_type: str,
    filing: dict,
):
    """
    Parse an EDGAR HTM filing and return validated Chunk objects.

    parse_html_filing() already chunks and assigns sections internally,
    so we skip the PDF pipeline's parser+chunker and go straight to
    Chunk construction → ChunkValidator.
    """
    from ingestion.chunk_validator import ChunkValidator
    from ingestion.html_parser import parse_html_filing
    from ingestion.models import Chunk

    logger.info("sec_edgar: HTM ingest pipeline for %s", doc_id)
    parsed = parse_html_filing(str(path), doc_id=doc_id, doc_type=doc_type)

    if not parsed:
        logger.error("sec_edgar: html_parser produced no pages for %s", doc_id)
        return []

    raw_chunks = [
        Chunk(
            doc_id=doc_id,
            chunk_index=idx,
            text=p.text,
            start_page=p.page_num,
            end_page=p.page_num,
            ticker=ticker.upper(),
            doc_type=doc_type,
            filing_date=filing["filing_date"],
            accession_number=filing["accession_number"],
            section=p.section,
        )
        for idx, p in enumerate(parsed)
    ]

    validated = ChunkValidator().validate(raw_chunks)
    logger.info(
        "sec_edgar: HTM parsed %d raw → %d valid chunks for %s",
        len(raw_chunks), len(validated), doc_id,
    )
    return validated
