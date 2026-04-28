"""
run_pipeline.py — unified CLI for StockRAG.

Subcommands:
  ingest  — parse, chunk, embed and upsert raw filings
  eval    — run end-to-end evaluation against eval_queries.json
  query   — retrieve chunks and generate a grounded answer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def _cmd_ingest(args: argparse.Namespace) -> None:
    from ingestion.batch_ingest import BatchIngester

    chunks_dir = Path("data/chunks")
    ingester = BatchIngester(data_dir=args.data_dir)
    processed = 0

    for doc_id, chunks in ingester.stream(overwrite=args.overwrite):
        if not chunks:
            continue
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out = chunks_dir / f"{doc_id}.json"
        if out.exists() and not args.overwrite:
            logger.info("Skipping (exists): %s", out.name)
        else:
            payload = {
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "chunks": [c.__dict__ for c in chunks],
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Wrote %d chunks → %s", len(chunks), out.name)
        processed += 1

    logger.info("Ingestion complete: %d document(s) processed", processed)
    if processed == 0:
        logger.warning("No documents processed — check %s or ingestion errors", args.data_dir)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _cmd_eval(args: argparse.Namespace) -> None:
    from evaluation.e2e_eval import run_eval

    query_ids = [q.strip() for q in args.query_ids.split(",")] if args.query_ids else None

    summary = run_eval(
        k=args.k,
        sample=args.sample,
        query_ids=query_ids,
    )
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def _cmd_query(args: argparse.Namespace) -> None:
    from vector_store.qdrant_client import get_qdrant_client, COLLECTION_NAME
    from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig
    from retrieval.retrieval import RetrievalPipeline, RetrievalConfig
    from retrieval.reranker import CrossEncoderReranker
    from generation.generator import generate_answer

    qdrant = get_qdrant_client()
    embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=64))
    pipeline = RetrievalPipeline(
        qdrant=qdrant,
        embedder=embedder,
        collection=COLLECTION_NAME,
        config=RetrievalConfig(top_k=args.top_k, fetch_k=50),
        reranker=CrossEncoderReranker(),
    )

    raw_results = pipeline.retrieve(query=args.query)
    chunks = [
        {
            "ticker": getattr(r, "ticker", None) or (r.metadata or {}).get("ticker"),
            "section": getattr(r, "section", None) or (r.metadata or {}).get("section"),
            "text": getattr(r, "text", None) or (r.metadata or {}).get("text"),
            "score": getattr(r, "score", 0.0),
            "chunk_id": getattr(r, "chunk_id", None),
        }
        for r in raw_results
    ]
    chunks = [c for c in chunks if c["ticker"] and c["section"] and c["text"]]

    if not chunks:
        logger.error("No valid chunks retrieved for query: %s", args.query)
        sys.exit(1)

    answer = generate_answer(query=args.query, retrieved_chunks=chunks)
    print(json.dumps(answer, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="StockRAG pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Parse, chunk, embed and upsert raw filings")
    p_ingest.add_argument("--data-dir", default="data/raw", help="Directory of raw PDFs/HTML")
    p_ingest.add_argument("--overwrite", action="store_true", help="Reprocess already-ingested files")

    # eval
    p_eval = sub.add_parser("eval", help="Run end-to-end evaluation")
    p_eval.add_argument("--query_ids", default=None, help="Comma-separated query IDs to run (e.g. Q07,Q13)")
    p_eval.add_argument("--sample", type=int, default=None, help="Run first N queries only")
    p_eval.add_argument("--k", type=int, default=5, help="Retrieval top-k")

    # query
    p_query = sub.add_parser("query", help="Retrieve and generate an answer for a single query")
    p_query.add_argument("query", help="The question to answer")
    p_query.add_argument("--top-k", type=int, default=10, dest="top_k", help="Retrieval top-k")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "query":
        _cmd_query(args)


if __name__ == "__main__":
    main()
