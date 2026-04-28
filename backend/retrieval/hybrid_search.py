from __future__ import annotations

import logging
from typing import Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridSearcher:
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        embedder,
        top_k: int = 10,
    ) -> None:
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.embedder = embedder
        self.top_k = top_k

        self._corpus: list[dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def build_index(self, ticker_filter: Optional[list[str]] = None) -> None:
        """Scroll Qdrant and build BM25 index over chunk texts."""
        ticker_set = {t.upper() for t in ticker_filter} if ticker_filter else None
        corpus: list[dict[str, Any]] = []
        offset = None

        while True:
            response = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = response

            for p in points:
                payload = p.payload or {}
                if ticker_set and payload.get("ticker", "").upper() not in ticker_set:
                    continue
                text = payload.get("text", "").strip()
                if not text:
                    continue
                corpus.append({
                    "id": str(p.id),
                    "text": text,
                    "ticker": payload.get("ticker", ""),
                    "section": payload.get("section", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "filing_date": payload.get("filing_date", ""),
                    "tokens": _tokenize(text),
                })

            if next_offset is None:
                break
            offset = next_offset

        self._corpus = corpus
        self._bm25 = BM25Okapi([doc["tokens"] for doc in corpus])
        logger.info("BM25 index built: %d documents", len(corpus))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        ticker_filter: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._corpus:
            raise RuntimeError("BM25 index not built — call build_index() first")

        k = top_k or self.top_k
        fetch_k = k * 3

        # Dense leg
        q_vec = self.embedder.embed_query(query)
        qdrant_filter = None
        if ticker_filter:
            from qdrant_client.models import FieldCondition, Filter, MatchAny
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="ticker",
                    match=MatchAny(any=[t.upper() for t in ticker_filter]),
                )]
            )

        response = self.qdrant.query_points(
            collection_name=self.collection,
            query=q_vec,
            query_filter=qdrant_filter,
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        )
        dense_ranked: dict[str, int] = {
            str(hit.id): rank + 1
            for rank, hit in enumerate(response.points)
        }

        # Sparse leg — restrict corpus to ticker if requested
        ticker_set = {t.upper() for t in ticker_filter} if ticker_filter else None
        if ticker_set:
            filtered_idx = [
                i for i, doc in enumerate(self._corpus)
                if doc["ticker"].upper() in ticker_set
            ]
            filtered_corpus = [self._corpus[i] for i in filtered_idx]
            bm25 = BM25Okapi([self._corpus[i]["tokens"] for i in filtered_idx])
        else:
            filtered_idx = list(range(len(self._corpus)))
            filtered_corpus = self._corpus
            bm25 = self._bm25

        q_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)
        sparse_ranked: dict[str, int] = {}
        for rank, idx in enumerate(
            sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
        ):
            doc_id = filtered_corpus[idx]["id"]
            sparse_ranked[doc_id] = rank + 1

        # RRF fusion
        all_ids = set(dense_ranked) | set(sparse_ranked)
        rrf_scores: dict[str, float] = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_ranked:
                score += 1.0 / (_RRF_K + dense_ranked[doc_id])
            if doc_id in sparse_ranked:
                score += 1.0 / (_RRF_K + sparse_ranked[doc_id])
            rrf_scores[doc_id] = score

        # Build id→payload lookup from dense hits + corpus
        payload_lookup: dict[str, dict[str, Any]] = {}
        for hit in response.points:
            payload_lookup[str(hit.id)] = hit.payload or {}
        for doc in filtered_corpus:
            if doc["id"] not in payload_lookup:
                payload_lookup[doc["id"]] = {
                    "ticker": doc["ticker"],
                    "section": doc["section"],
                    "text": doc["text"],
                    "doc_type": doc["doc_type"],
                    "filing_date": doc["filing_date"],
                }

        results: list[dict[str, Any]] = []
        for doc_id, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            p = payload_lookup.get(doc_id, {})
            results.append({
                "id": doc_id,
                "text": p.get("text", ""),
                "ticker": p.get("ticker", ""),
                "section": p.get("section", ""),
                "doc_type": p.get("doc_type", ""),
                "filing_date": p.get("filing_date", ""),
                "score": rrf,
                "rrf_score": round(rrf, 6),
                "dense_rank": dense_ranked.get(doc_id),
                "sparse_rank": sparse_ranked.get(doc_id),
            })

        return results
