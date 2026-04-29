"""
scripts/ingest_batch.py — Batch-ingest a folder of PDF/HTML filings.

Filename convention (required):
    TICKER_YEAR_DOCTYPE.{pdf,html,htm}
    e.g. AAPL_2024_10K.pdf  → ticker=AAPL, filing_date=YEAR-01-01, doc_type=10-K
         AAPL_2024_10K.html → same metadata, parsed via HTMLParser

Usage:
    python scripts/ingest_batch.py --dir data/raw/filings [--dry-run] [--collection sec_filings]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_batch")

# TICKER_YEAR_DOCTYPE.pdf — DOCTYPE may contain alphanumerics and hyphens
_FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<year>\d{4})_(?P<dtype>[A-Z0-9\-]+)$",
    re.IGNORECASE,
)

_DOCTYPE_MAP = {
    "10K":  "10-K",
    "10-K": "10-K",
    "10Q":  "10-Q",
    "10-Q": "10-Q",
    "8K":   "8-K",
    "8-K":  "8-K",
    "EARNINGS": "earnings_transcript",
    "TRANSCRIPT": "earnings_transcript",
}


def _parse_filename(stem: str) -> tuple[str, str, str] | None:
    """
    Returns (ticker, filing_date, doc_type) or None if stem doesn't match.
    filing_date is YEAR-01-01 — best available without reading the PDF.
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        return None

    ticker = m.group("ticker").upper()
    year = m.group("year")
    raw_dtype = m.group("dtype").upper()

    doc_type = _DOCTYPE_MAP.get(raw_dtype)
    if doc_type is None:
        return None

    return ticker, f"{year}-01-01", doc_type


_HTML_EXTENSIONS = {".html", ".htm"}


def _ingest_one(
    file_path: Path,
    ticker: str,
    filing_date: str,
    doc_type: str,
    collection: str,
    dry_run: bool,
) -> int:
    """Run the full ingest pipeline for one file. Returns chunk count."""
    from ingestion.doc_metadata import DocumentMetadata
    from ingestion.ingest import Ingest

    doc_id = f"{ticker}_{filing_date[:4]}_{doc_type.replace('-', '')}"

    metadata = DocumentMetadata(
        ticker=ticker,
        doc_type=doc_type,
        filing_date=filing_date,
        accession_number="unknown",
    )

    if file_path.suffix.lower() in _HTML_EXTENSIONS:
        from ingestion.html_parser import HTMLParser
        parser = HTMLParser()
    else:
        parser = None  # Ingest defaults to PDFParser

    pipeline = Ingest(doc_id=doc_id, metadata=metadata, parser=parser)
    chunks = pipeline.run(str(file_path))

    if not chunks:
        raise ValueError("No chunks produced — check PDF content")

    if dry_run:
        return len(chunks)

    from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig
    from vector_store.qdrant_client import get_qdrant_client
    from vector_store.qdrant_store import QdrantStore

    chunk_dicts = [
        {
            "chunk_id":         f"{c.doc_id}_{c.chunk_index}",
            "doc_id":           c.doc_id,
            "ticker":           c.ticker or ticker,
            "doc_type":         c.doc_type or doc_type,
            "filing_date":      c.filing_date or filing_date,
            "accession_number": c.accession_number or "unknown",
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

    qdrant = get_qdrant_client()
    store = QdrantStore(client=qdrant, collection_name=collection)
    store.upsert_chunks(embedded)

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingest_batch",
        description="Batch-ingest a folder of PDF filings.",
    )
    parser.add_argument("--dir",        required=True, help="Folder to scan recursively for .pdf / .html / .htm files")
    parser.add_argument("--collection", default="sec_filings", help="Qdrant collection name")
    parser.add_argument("--dry-run",    action="store_true", help="Parse and chunk only — no embedding or upsert")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        logger.error("Directory not found: %s", root)
        sys.exit(1)
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        sys.exit(1)

    files = sorted(
        f for ext in ("*.pdf", "*.html", "*.htm")
        for f in root.rglob(ext)
    )
    if not files:
        logger.warning("No .pdf / .html / .htm files found under %s", root)
        sys.exit(0)

    logger.info("Found %d file(s) under %s", len(files), root)

    ingested_files = 0
    total_chunks = 0
    skipped = 0

    for filing in files:
        parsed = _parse_filename(filing.stem)
        if parsed is None:
            logger.warning(
                "Skipping %s — filename does not match TICKER_YEAR_DOCTYPE pattern",
                filing.name,
            )
            skipped += 1
            continue

        ticker, filing_date, doc_type = parsed
        label = filing.name

        print(f"Ingesting {label}...", end=" ", flush=True)
        try:
            n_chunks = _ingest_one(
                file_path=filing,
                ticker=ticker,
                filing_date=filing_date,
                doc_type=doc_type,
                collection=args.collection,
                dry_run=args.dry_run,
            )
            print(f"done ({n_chunks} chunks)")
            ingested_files += 1
            total_chunks += n_chunks
        except Exception as exc:
            print("FAILED")
            logger.error("Failed to ingest %s: %s", filing.name, exc)
            skipped += 1

    suffix = " [dry run]" if args.dry_run else ""
    print(
        f"\nIngested {ingested_files} file(s), {total_chunks} chunks total, "
        f"{skipped} skipped{suffix}"
    )


if __name__ == "__main__":
    main()
