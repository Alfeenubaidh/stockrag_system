"""
evaluation/eval_runner.py — Retrieval quality evaluation for StockRAG.

Loads eval_queries.json, runs the live retrieval pipeline, computes
hit@3, MRR, and multi-ticker coverage per query, then prints a summary.

Usage:
    python -m evaluation.eval_runner
    python -m evaluation.eval_runner --queries evaluation/eval_queries.json

Query format (eval_queries.json):
    {
      "query":   str,
      "ticker":  str,           # single-ticker queries
      "tickers": list[str],     # multi-ticker / comparison queries
      "section": str            # expected section (substring match)
    }
    One of "ticker" or "tickers" must be present.

Dependencies:
    Qdrant must be running at localhost:6333 with collection "sec_filings"
    populated before running this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig
from vector_store.qdrant_client import get_qdrant_client
from retrieval.retrieval import RetrievalPipeline, RetrievalConfig
from retrieval.reranker import CrossEncoderReranker
from evaluation.eval import hit_at_k, mean_reciprocal_rank, coverage
from observability.pipeline_observer import observer

logging.basicConfig(
    level=logging.WARNING,          # suppress retrieval debug noise
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

HIT_K = 3


# ---------------------------------------------------------------------------
# Pipeline init — mirrors eval_runner_phase2.py
# ---------------------------------------------------------------------------

def _build_pipeline() -> RetrievalPipeline:
    qdrant = get_qdrant_client()
    embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=64))
    return RetrievalPipeline(
        qdrant=qdrant,
        embedder=embedder,
        collection="sec_filings",
        config=RetrievalConfig(top_k=10),
        reranker=CrossEncoderReranker(),
    )


# ---------------------------------------------------------------------------
# Query loader
# ---------------------------------------------------------------------------

def _load_queries(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for i, entry in enumerate(raw):
        if "query" not in entry:
            raise ValueError(f"Entry {i} missing 'query' field")
        exp = entry.get("expected", {})
        # Normalise to flat structure regardless of whether fields are nested
        tickers = entry.get("tickers") or entry.get("ticker") or exp.get("tickers", [])
        if isinstance(tickers, str):
            tickers = [tickers]
        sections = entry.get("sections") or exp.get("sections", [])
        if isinstance(sections, str):
            sections = [sections]
        section = entry.get("section") or (sections[0] if sections else "")
        keywords = entry.get("keywords") or exp.get("keywords", [])
        doc_type = entry.get("doc_type") or exp.get("doc_type")
        if not tickers:
            raise ValueError(f"Entry {i} has no tickers")
        queries.append({
            "query": entry["query"],
            "tickers": tickers,
            "section": section,
            "sections": sections,
            "keywords": keywords,
            "doc_type": doc_type,
        })
    return queries


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def _eval_query(pipeline: RetrievalPipeline, entry: dict) -> dict:
    query = entry["query"]
    section = entry["section"]
    expected_tickers: list[str] = entry["tickers"]
    is_multi = len(expected_tickers) > 1

    sections = entry.get("sections") or [section]
    doc_type = entry.get("doc_type")
    results = pipeline.retrieve(query, tickers=expected_tickers, doc_type=doc_type)

    h = hit_at_k(results, expected_tickers, sections, k=HIT_K)
    mrr = mean_reciprocal_rank(results, expected_tickers, sections)
    cov = coverage(results, expected_tickers) if is_multi else None

    return {
        "query": query,
        "tickers": expected_tickers,
        "section": section,
        "is_multi": is_multi,
        "n_results": len(results),
        f"hit@{HIT_K}": h,
        "mrr": mrr,
        "coverage": cov,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(queries_path: Path) -> None:
    print(f"\nStockRAG Retrieval Eval")
    print(f"Queries : {queries_path}")
    print(f"Hit@K   : {HIT_K}\n")

    queries = _load_queries(queries_path)
    print(f"Loaded {len(queries)} queries. Initialising pipeline...")
    pipeline = _build_pipeline()
    print("Pipeline ready.\n")

    records: list[dict] = []
    for i, entry in enumerate(queries, start=1):
        observer.start_trace(query_id=f"Q{i:02d}", query_text=entry["query"])
        rec = _eval_query(pipeline, entry)
        observer.flush_trace(status="ok")
        records.append(rec)

        hit_flag    = "HIT " if rec[f"hit@{HIT_K}"] else "MISS"
        cov_str     = f"  coverage={rec['coverage']:.2f}" if rec["is_multi"] else ""
        tickers_str = "/".join(rec["tickers"])
        print(
            f"  [{hit_flag}] Q{i:02d} {tickers_str:<12} "
            f"mrr={rec['mrr']:.3f}  "
            f"results={rec['n_results']}"
            f"{cov_str}"
        )

    # Aggregate
    n = len(records)
    avg_hit  = sum(1 for r in records if r[f"hit@{HIT_K}"]) / n if n else 0.0
    avg_mrr  = sum(r["mrr"] for r in records) / n if n else 0.0

    multi = [r for r in records if r["is_multi"]]
    avg_cov = (
        sum(r["coverage"] for r in multi) / len(multi)
        if multi else None
    )

    print()
    print("-" * 50)
    print(f"  Queries    : {n}")
    print(f"  avg hit@{HIT_K}  : {avg_hit:.3f}  ({sum(1 for r in records if r[f'hit@{HIT_K}'])}/{n})")
    print(f"  avg MRR    : {avg_mrr:.3f}")
    if avg_cov is not None:
        print(f"  avg coverage (multi-ticker): {avg_cov:.3f}  (n={len(multi)})")
    print("-" * 50 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="StockRAG retrieval eval runner")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("evaluation/eval_queries.json"),
    )
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"ERROR: queries file not found: {args.queries}", file=sys.stderr)
        sys.exit(1)

    run_eval(args.queries)


if __name__ == "__main__":
    main()
