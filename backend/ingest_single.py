"""
ingest_single.py — Ingest a single PDF filing into the StockRAG vector store.

Usage:
    python ingest_single.py --ticker AAPL --file path/to/aapl_10k.pdf \
        --doc-type 10-K --filing-date 2024-09-28 [--accession 0000320193-24-000123]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_single")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingest_single",
        description="Ingest a single PDF filing into the StockRAG vector store.",
    )
    parser.add_argument("--ticker",       required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--file",         required=True, help="Path to the PDF file")
    parser.add_argument("--doc-type",     required=True, dest="doc_type",
                        help="Document type: 10-K, 10-Q, 8-K, earnings_transcript")
    parser.add_argument("--filing-date",  required=True, dest="filing_date",
                        help="Filing date in ISO format: YYYY-MM-DD")
    parser.add_argument("--accession",    default="unknown", dest="accession_number",
                        help="SEC accession number (optional)")
    parser.add_argument("--collection",        default="sec_filings", help="Qdrant collection name")
    parser.add_argument("--exclude-sections",  default="Exhibits",
                        dest="exclude_sections",
                        help="Comma-separated section names to drop before embedding (default: Exhibits)")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Parse and chunk only — do not embed or upsert")
    args = parser.parse_args()

    pdf_path = Path(args.file)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        logger.error("Only PDF files are supported, got: %s", pdf_path.suffix)
        sys.exit(1)

    ticker = args.ticker.upper()
    doc_id = f"{ticker}_{args.filing_date[:4]}_{args.doc_type.replace('-', '')}"
    logger.info("doc_id=%s  file=%s", doc_id, pdf_path)

    # 1. Ingest (parse → clean → chunk → validate → section detect)
    from ingestion.doc_metadata import DocumentMetadata
    from ingestion.ingest import Ingest

    metadata = DocumentMetadata(
        ticker=ticker,
        doc_type=args.doc_type,
        filing_date=args.filing_date,
        accession_number=args.accession_number,
    )

    pipeline = Ingest(doc_id=doc_id, metadata=metadata)
    chunks = pipeline.run(str(pdf_path))

    logger.info("Ingestion complete: %d chunks produced", len(chunks))

    if not chunks:
        logger.error("No chunks produced — check the PDF and section detector")
        sys.exit(1)

    # Filter excluded sections
    exclude = {s.strip() for s in args.exclude_sections.split(",") if s.strip()}
    if exclude:
        before = len(chunks)
        chunks = [c for c in chunks if (c.section or "") not in exclude]
        dropped = before - len(chunks)
        if dropped:
            logger.info("Excluded %d chunk(s) from sections: %s", dropped, ", ".join(sorted(exclude)))

    if not chunks:
        logger.error("No chunks remain after section exclusion")
        sys.exit(1)

    if args.dry_run:
        from collections import Counter
        section_counts = Counter(c.section for c in chunks)
        print(f"\nChunks after exclusion: {len(chunks)}")
        for section, count in sorted(section_counts.items(), key=lambda x: -(x[1] or 0)):
            print(f"  {count:3d}  {section}")
        print()
        for c in chunks[:5]:
            print(f"  [{c.section}] {c.text[:80]}...")
        logger.info("Dry run complete — nothing written to Qdrant")
        return

    # 2. Embed
    from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig

    embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=128))
    chunk_dicts = [
        {
            "chunk_id":         f"{c.doc_id}_{c.chunk_index}",
            "doc_id":           c.doc_id,
            "ticker":           c.ticker or ticker,
            "doc_type":         c.doc_type or args.doc_type,
            "filing_date":      c.filing_date or args.filing_date,
            "accession_number": c.accession_number or args.accession_number,
            "section":          c.section or "unknown",
            "start_page":       c.start_page,
            "end_page":         c.end_page,
            "chunk_index":      c.chunk_index,
            "text":             c.text,
        }
        for c in chunks
    ]

    embedded = embedder.embed_chunks(chunk_dicts)
    logger.info("Embedding complete: %d vectors", len(embedded))

    # 3. Upsert
    from vector_store.qdrant_client import get_qdrant_client
    from vector_store.qdrant_store import QdrantStore

    qdrant = get_qdrant_client()
    store = QdrantStore(client=qdrant, collection_name=args.collection)
    store.upsert_chunks(embedded)

    logger.info("Upsert complete: %d chunks written to collection '%s'", len(embedded), args.collection)


if __name__ == "__main__":
    main()
