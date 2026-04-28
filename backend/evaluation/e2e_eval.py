from __future__ import annotations
print("RUNNING FILE:", __file__)

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Resolve package root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from evaluation.scorers.retrieval_scorer import score_retrieval
from evaluation.scorers.generation_scorer import score_generation
from evaluation.scorers.e2e_scorer import score_e2e
from generation.generator import generate_answer
from observability.pipeline_observer import observer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy third-party debug output; keep retrieval pipeline visible
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger("e2e_eval")

_EVAL_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
_DEFAULT_K = 5


# -----------------------------
# LOAD QUERIES
# -----------------------------
def _load_eval_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("eval_queries.json must be a list")

    return data


# -----------------------------
# RETRIEVAL WRAPPER
# -----------------------------
def _retrieve(query: str, pipeline: Any, k: int):
    if hasattr(pipeline, "retrieve"):
        return pipeline.retrieve(query=query)
    elif hasattr(pipeline, "query"):
        return pipeline.query(query, top_k=k)
    else:
        raise RuntimeError("Pipeline has no retrieve/query method")


# -----------------------------
# RESULT → CHUNK NORMALIZATION
# -----------------------------
def _result_to_chunk(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        payload = result.get("payload", result)

        return {
            "ticker": payload.get("ticker"),
            "section": payload.get("section"),
            "text": payload.get("text"),
            "score": payload.get("score", 0.0),
            "chunk_id": payload.get("chunk_id"),
        }

    # 🔥 FIX: pull directly from object if metadata missing
    payload = getattr(result, "metadata", {}) or {}

    return {
        "ticker": payload.get("ticker") or getattr(result, "ticker", None),
        "section": payload.get("section") or getattr(result, "section", None),
        "text": payload.get("text") or getattr(result, "text", None),
        "score": payload.get("score", getattr(result, "score", 0.0)),
        "chunk_id": payload.get("chunk_id") or getattr(result, "chunk_id", None),
    }

# -----------------------------
# HARD VALIDATION
# -----------------------------
def _validate_chunks(chunks: List[Dict[str, Any]], idx: int) -> List[Dict[str, Any]]:
    valid = []

    for c in chunks:
        if not c.get("ticker"):
            print(f"[{idx}] [FAIL] Missing ticker")
            continue
        if not c.get("section"):
            print(f"[{idx}] [FAIL] Missing section")
            continue
        if not c.get("text"):
            print(f"[{idx}] [FAIL] Missing text")
            continue

        valid.append(c)

    return valid


# -----------------------------
# BUILD PIPELINE
# -----------------------------
def _build_pipeline():
    try:
        from vector_store.qdrant_client import get_qdrant_client, COLLECTION_NAME
        from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig
        from retrieval.retrieval import RetrievalPipeline, RetrievalConfig
        from retrieval.reranker import CrossEncoderReranker

        qdrant = get_qdrant_client()
        embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=64))

        return RetrievalPipeline(
            qdrant=qdrant,
            embedder=embedder,
            collection=COLLECTION_NAME,
            config=RetrievalConfig(top_k=10, fetch_k=50),
            reranker=CrossEncoderReranker(),
        )

    except Exception as exc:
        raise RuntimeError("Failed to initialize pipeline") from exc


# -----------------------------
# MAIN EVAL LOOP
# -----------------------------
def run_eval(
    eval_queries_path: Path = _EVAL_QUERIES_PATH,
    k: int = _DEFAULT_K,
    sample: int | None = None,
    query_ids: List[str] | None = None,
) -> Dict[str, Any]:

    queries = _load_eval_queries(eval_queries_path)
    if query_ids is not None:
        id_set = set(query_ids)
        queries = [q for q in queries if q.get("id") in id_set]
        logger.info("--query_ids filter: running %d queries", len(queries))
    elif sample is not None:
        total = len(queries)
        queries = queries[:sample]
        logger.info("--sample %d: running %d of %d queries", sample, len(queries), total)
    pipeline = _build_pipeline()

    r_scores_all = []
    g_scores_all = []
    e2e_scores_all = []

    failure_counts = {
        "retrieval": 0,
        "generation": 0,
        "generation_crash": 0,
        "both": 0,
        "pass": 0,
    }

    for idx, item in enumerate(queries, start=1):
        query = item["query"]
        query_id = item.get("id", f"q{idx}")
        expected = item.get("expected", {})

        print("\n" + "=" * 60)
        print(f"[{idx}] QUERY: {query}")

        generation_crashed = False
        observer.start_trace(query_id=query_id, query_text=query)

        # -----------------------------
        # RETRIEVAL
        # -----------------------------
        try:
            raw_results = _retrieve(query, pipeline, k)
        except Exception as e:
            print(f"[{idx}] [FAIL] Retrieval crashed: {e}")
            failure_counts["both"] += 1
            observer.flush_trace(status="error")
            continue

        if not raw_results:
            print(f"[{idx}] [FAIL] EMPTY RETRIEVAL")
            failure_counts["both"] += 1
            observer.flush_trace(status="error")
            continue

        # DEBUG SAMPLE
        print(f"[{idx}] Raw result sample:", raw_results[0])

        # -----------------------------
        # NORMALIZE + VALIDATE
        # -----------------------------
        chunks = [_result_to_chunk(r) for r in raw_results]
        chunks = _validate_chunks(chunks, idx)

        if not chunks:
            print(f"[{idx}] [FAIL] No valid chunks")
            failure_counts["both"] += 1
            observer.flush_trace(status="error")
            continue

        print(f"[{idx}] Valid chunk sample:", chunks[0])

        # -----------------------------
        # RETRIEVAL SCORING
        # -----------------------------
        r_scores = score_retrieval(chunks, expected, k=k)

        # -----------------------------
        # GENERATION (optional)
        # -----------------------------
        try:
            output = generate_answer(query, chunks)
            print(f"[{idx}] Output:", output)

            if isinstance(output, dict) and output.get("answers"):
                g_scores = score_generation(output, chunks, expected)
            else:
                print(f"[{idx}] [WARN] Empty or invalid generation output")
                g_scores = {"score": 0.0}

        except Exception as e:
            print(f"[{idx}] [WARN] Generator skipped: {e}")
            generation_crashed = True
            output = None
            g_scores = {"score": 0.0}

        if not generation_crashed and not (isinstance(output, dict) and output.get("answers")):
            # Validation failure: dict returned with error key but empty answers — not a crash
            if isinstance(output, dict) and "error" in output:
                generation_crashed = False
            else:
                generation_crashed = True

        # -----------------------------
        # E2E
        # -----------------------------
        e2e = score_e2e(r_scores["score"], g_scores["score"])

        print(
            f"[{idx}] R={r_scores['score']} | G={g_scores['score']} | "
            f"E2E={e2e['E2E']} | {e2e['failure_type']}"
        )

        r_scores_all.append(r_scores["score"])
        g_scores_all.append(g_scores["score"])
        e2e_scores_all.append(e2e["E2E"])

        effective_failure_type = "generation_crash" if generation_crashed else e2e["failure_type"]

        if generation_crashed:
            failure_counts["generation_crash"] += 1
        else:
            failure_counts[e2e["failure_type"]] += 1

        observer.log_eval_scores(
            r_score=r_scores["score"],
            g_score=g_scores["score"],
            e2e_score=e2e["E2E"],
            failure_type=effective_failure_type,
            generation_crashed=generation_crashed,
        )
        observer.flush_trace(status="ok")

    # -----------------------------
    # SUMMARY
    # -----------------------------
    n = max(len(queries), 1)

    summary = {
        "avg_R": round(sum(r_scores_all) / n, 4),
        "avg_G": round(sum(g_scores_all) / n, 4),
        "avg_E2E": round(sum(e2e_scores_all) / n, 4),
        "failures": failure_counts,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    return summary


# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=_EVAL_QUERIES_PATH)
    parser.add_argument("--k", type=int, default=_DEFAULT_K)
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Limit evaluation to the first N queries")
    parser.add_argument("--query_ids", type=str, default=None, metavar="IDS",
                        help="Comma-separated list of query IDs to run (e.g. Q01,Q03)")

    args = parser.parse_args()

    query_ids = args.query_ids.split(",") if args.query_ids else None
    run_eval(eval_queries_path=args.queries, k=args.k, sample=args.sample, query_ids=query_ids)