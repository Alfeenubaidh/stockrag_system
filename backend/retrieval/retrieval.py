from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, MatchText,
)

from embeddings.embedder import EmbeddingPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.query_parser import QueryParser
from retrieval.ranking_signals import RankingSignalScorer
from retrieval.hybrid_search import HybridSearcher
from observability.pipeline_observer import observer

logger = logging.getLogger(__name__)

FALLBACK_K = 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    top_k: int = 5
    fetch_k: int = 50
    score_threshold: float = 0.0
    use_mmr: bool = False
    use_hybrid: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    doc_id: str
    section: str
    doc_type: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        ticker = self.metadata.get("ticker", "?")
        return f"[{self.score:.3f}] {ticker}/{self.section[:30]} {self.text[:80]}..."


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: EmbeddingPipeline,
        collection: str = "sec_filings",
        config: RetrievalConfig = RetrievalConfig(),
        reranker: Optional[CrossEncoderReranker] = None,
        rewriter=None,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection
        self.config = config
        self.reranker = reranker
        self.parser = QueryParser()
        self.signal_scorer = RankingSignalScorer()
        self._hybrid: Optional[HybridSearcher] = (
            HybridSearcher(qdrant, collection, embedder, top_k=config.fetch_k)
            if config.use_hybrid else None
        )

    # ------------------------------------------------------------------

    def rebuild_hybrid_index(self, ticker_filter: Optional[List[str]] = None) -> None:
        if self._hybrid is None:
            logger.warning("rebuild_hybrid_index called but use_hybrid=False — skipping")
            return
        self._hybrid.build_index(ticker_filter=ticker_filter)

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        tickers: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        doc_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:

        effective_k = top_k if top_k is not None else self.config.top_k

        # -------------------------------
        # 1. Parse query
        # -------------------------------
        pq = self.parser.parse(query)

        if tickers is not None:
            # Explicit caller override — always respected regardless of count
            active_tickers = tickers
        elif len(pq.tickers) >= 1:
            # One or more parser-detected tickers — always filter to prevent
            # cross-ticker contamination from other companies in the index
            active_tickers = pq.tickers
        else:
            # No ticker detected — cast wide
            active_tickers = None

        observer.log_parsing(
            original_query=query,
            expanded_queries=pq.expanded_queries or [],
            inferred_ticker=pq.inferred_ticker,
            confidence=pq.confidence,
            section_hint=pq.section_hint,
            active_tickers=active_tickers or [],
        )

        # -------------------------------
        # 2. Build filter
        # -------------------------------
        q_filter = _build_filter(
            tickers=active_tickers,
            doc_type=doc_type,
        )
        logger.info(
            "[TICKER_FILTER] caller_tickers=%r  parser_tickers=%r  active_tickers=%r  filter=%r",
            tickers, pq.tickers, active_tickers, q_filter,
        )

        # -------------------------------
        # 3. Vector search (multi-query) or hybrid search
        # -------------------------------
        queries_to_run = [e for e in (pq.expanded_queries or [query]) if len(e.strip()) >= 20]
        if not queries_to_run:
            queries_to_run = [query]
        logger.info("[DEBUG] queries_sent=%s", queries_to_run)
        merged: Dict[str, RetrievalResult] = {}

        if self.config.use_hybrid and self._hybrid is not None:
            hybrid_hits = self._hybrid.search(
                query=query,
                ticker_filter=active_tickers,
                top_k=self.config.fetch_k,
            )
            for hit in hybrid_hits:
                result = RetrievalResult(
                    chunk_id=hit["id"],
                    score=hit["rrf_score"],
                    text=hit["text"],
                    doc_id=hit.get("doc_id", ""),
                    section=_normalize_section(hit.get("section", "")),
                    doc_type=hit.get("doc_type", ""),
                    metadata={
                        "ticker": hit.get("ticker"),
                        "filing_date": _resolve_filing_date(
                            hit.get("filing_date", ""), hit.get("doc_id", "")
                        ),
                        "fallback": False,
                    },
                )
                merged[result.chunk_id] = result
            logger.info("[HYBRID] merged %d results", len(merged))
            if not merged:
                return []
            candidates = sorted(merged.values(), key=lambda r: r.score, reverse=True)[: self.config.fetch_k]
            # skip straight to reranking
            observer.log_retrieval(queries_sent=queries_to_run, filter_applied=str(q_filter) if q_filter else None, results=candidates)
            for r in candidates:
                bonus, penalty = self.signal_scorer.score(
                    query=query, chunk_text=r.text, chunk_ticker=r.metadata.get("ticker"),
                    inferred_ticker=pq.inferred_ticker, confidence=pq.confidence,
                    section_hint=pq.section_hint, chunk_section=r.section,
                )
                r.score = r.score + bonus - penalty
            candidates = [r for r in candidates if r.text and len(r.text.strip()) >= 50]
            if not candidates:
                return []
            reranked = self.reranker.rerank(query=query, results=candidates, top_k=effective_k, section_hint=pq.section_hint) if self.reranker else candidates[:effective_k]
            return reranked or candidates[:FALLBACK_K]

        def _run_search(q_text, filt, limit, dest):
            q_vec = self.embedder.embed_query(q_text)
            logger.info(
                "[QDRANT_CALL] collection=%r  query=%r  filter=%r  limit=%d",
                self.collection, q_text[:80], filt, limit,
            )

            response = self.qdrant.query_points(
                collection_name=self.collection,
                query=q_vec,
                query_filter=filt,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            total_hits = len(response.points)
            short_dropped = 0
            for hit in response.points:
                result = _hit_to_result(hit)
                if not result.text or len(result.text.strip()) < 50:
                    short_dropped += 1
                    continue
                existing = dest.get(result.chunk_id)
                if existing is None or result.score > existing.score:
                    dest[result.chunk_id] = result

            returned_tickers = [hit.payload.get("ticker") for hit in response.points if hit.payload]
            logger.info(
                "[QDRANT_RESULT] hits=%d  short_dropped=%d  tickers_returned=%s",
                total_hits, short_dropped, returned_tickers,
            )

        for q_text in queries_to_run:
            _run_search(q_text, q_filter, self.config.fetch_k, merged)

        if not merged:
            return []

        candidates = sorted(
            merged.values(), key=lambda r: r.score, reverse=True
        )[: self.config.fetch_k]

        observer.log_retrieval(
            queries_sent=queries_to_run,
            filter_applied=str(q_filter) if q_filter else None,
            results=candidates,
        )

        # -------------------------------
        # 4. APPLY RANKING SIGNALS (CRITICAL)
        # -------------------------------
        for r in candidates:
            bonus, penalty = self.signal_scorer.score(
                query=query,
                chunk_text=r.text,
                chunk_ticker=r.metadata.get("ticker"),
                inferred_ticker=pq.inferred_ticker,
                confidence=pq.confidence,
                section_hint=pq.section_hint,
                chunk_section=r.section,
            )
            r.score = r.score + bonus - penalty

        pre_filter = len(candidates)
        candidates = [r for r in candidates if r.text and len(r.text.strip()) >= 50]
        post_filter = len(candidates)

        logger.debug(
            "post_signal_filter  query=%r  before=%d  after=%d  dropped=%d",
            query[:60], pre_filter, post_filter, pre_filter - post_filter,
        )

        if not candidates:
            logger.warning(
                "no_candidates_after_filter  query=%r  pre_filter=%d",
                query[:80], pre_filter,
            )
            return []

        if post_filter < effective_k:
            logger.warning(
                "candidate_shortfall  query=%r  post_filter=%d  effective_k=%d",
                query[:60], post_filter, effective_k,
            )

        # -------------------------------
        # 5. Cross-encoder rerank
        # -------------------------------
        if self.reranker:
            try:
                reranked = self.reranker.rerank(
                    query=query,
                    results=candidates,
                    top_k=effective_k,
                    section_hint=pq.section_hint,
                )
            except Exception as e:
                logger.error(f"Reranker failed: {e}")
                reranked = candidates[:effective_k]
        else:
            reranked = candidates[:effective_k]

        if not reranked:
            return candidates[:FALLBACK_K]

        logger.info("[DEBUG] top_chunks=%s", [(r.metadata.get("ticker"), r.section, round(r.score, 3)) for r in reranked])

        # -------------------------------
        # 6. Ticker consistency rerank
        # -------------------------------
        ticker_scores = defaultdict(float)

        for r in reranked:
            ticker_scores[r.metadata.get("ticker")] += r.score

        dominant = max(ticker_scores, key=ticker_scores.get)

        reranked = sorted(
            reranked,
            key=lambda r: (r.metadata.get("ticker") == dominant, r.score),
            reverse=True
        )

        return reranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filter(
    tickers: Optional[List[str]],
    doc_type: Optional[str],
    section: Optional[str] = None,
) -> Optional[Filter]:

    must = []

    if tickers:
        must.append(
            FieldCondition(
                key="ticker",
                match=MatchAny(any=[t.upper() for t in tickers]),
            )
        )

    if doc_type:
        must.append(
            FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
        )

    if section:
        must.append(
            FieldCondition(key="section", match=MatchText(text=section))
        )

    return Filter(must=must) if must else None


_SECTION_LABELS: dict[str, str] = {
    "md&a": "MD&A",
    "risk_factors": "Risk Factors",
    "risk factors": "Risk Factors",
    "financial_statements": "Financial Statements",
    "financial statements": "Financial Statements",
    "business": "Business",
    "exhibits": "Exhibits",
    "legal proceedings": "Legal Proceedings",
    "legal_proceedings": "Legal Proceedings",
    "market risk": "Market Risk",
    "market_risk": "Market Risk",
    "cybersecurity": "Cybersecurity",
    "controls": "Controls",
    "ownership": "Ownership",
    "properties": "Properties",
    "changes": "Changes",
}

_YEAR_RE = re.compile(r"(\d{4})")


def _resolve_filing_date(filing_date: str, doc_id: str) -> str:
    if filing_date and filing_date not in ("unknown", ""):
        return filing_date
    m = _YEAR_RE.search(doc_id)
    return m.group(1) if m else ""


def _normalize_section(slug: str) -> str:
    return _SECTION_LABELS.get(slug.lower(), slug)


def _hit_to_result(hit) -> RetrievalResult:
    p = hit.payload or {}
    doc_id = p.get("doc_id", "")
    filing_date = _resolve_filing_date(p.get("filing_date", ""), doc_id)
    section = _normalize_section(p.get("section", ""))

    return RetrievalResult(
        chunk_id=str(hit.id),
        score=float(hit.score),
        text=p.get("text", ""),
        doc_id=doc_id,
        section=section,
        doc_type=p.get("doc_type", ""),
        metadata={
            "ticker": p.get("ticker"),
            "filing_date": filing_date,
            "fallback": False,
        },
    )