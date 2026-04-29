"""
reranker.py — Cross-encoder reranking with section bonus and ranking signals.

Final scoring formula per chunk:
    final = ce_score
          + section_bonus        (if section_hint matches result.section)
          + keyword_bonus        (informative query term overlap + co-occurrence)
          - boilerplate_penalty  (fraction of text that is generic SEC filler)

Weights are additive so each component acts as a tiebreaker at its scale:
  - CE score range on 10-K text: roughly [-8, +5]
  - section_bonus: 3.5 (overrides same-section CE variance)
  - keyword_bonus: up to ~3.5 (overrides generic vs. specific chunk gap)
  - boilerplate_penalty: up to 1.5 (demotes filler-heavy chunks)
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    from sentence_transformers import CrossEncoder
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

from retrieval.ranking_signals import RankingSignalScorer
from observability.pipeline_observer import observer

logger = logging.getLogger(__name__)

if not _RERANKER_AVAILABLE:
    logger.warning(
        "sentence-transformers not installed — CrossEncoderReranker disabled. "
        "Install torch + sentence-transformers to enable reranking."
    )

_SECTION_BONUS = 6.0
_signal_scorer = RankingSignalScorer()


# SEC filings split legal/regulatory risk across Item 1A (risk factors) and
# Item 3 (legal proceedings); treat them as interchangeable for reranking.
_SECTION_ALIASES: dict[str, set[str]] = {
    "riskfactors": {"legalproceedings"},
    "legalproceedings": {"riskfactors"},
}


def _section_matches(hint: str, section: str) -> bool:
    if not hint or not section:
        return False
    h = hint.lower().replace(" ", "").replace("_", "").replace("-", "")
    s = section.lower().replace(" ", "").replace("_", "").replace("-", "")
    if h in s or s in h:
        return True
    return s in _SECTION_ALIASES.get(h, set())


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        if _RERANKER_AVAILABLE:
            self.model = CrossEncoder(model_name)
        else:
            self.model = None

    def rerank(
        self,
        query: str,
        results: List,
        top_k: int,
        section_hint: Optional[str] = None,
    ) -> List:
        """
        Rerank results using cross-encoder + section bonus + keyword signals.

        Args:
            query:        Original user query (full, untruncated).
            results:      List of RetrievalResult objects.
            top_k:        Number of results to return.
            section_hint: Inferred section string (e.g. "risk factors", "gaming").
        """
        if not results:
            return results

        if self.model is None:
            logger.debug("rerank: no model available, returning top_k by existing score")
            return results[:top_k]

        # Cross-encoder scores full chunk text — no truncation
        pairs  = [(query, r.text) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        rerank_hits: list[dict] = []
        for r, ce_score in zip(results, scores):
            # 1. Section relevance bonus
            section_bonus = _SECTION_BONUS if _section_matches(section_hint, r.section) else 0.0

            # 2. Keyword overlap + co-occurrence bonus; boilerplate penalty
            keyword_bonus, boilerplate_penalty = _signal_scorer.score(query, r.text)

            r.score = float(ce_score) + section_bonus + keyword_bonus - boilerplate_penalty

            logger.debug(
                "chunk=%s ce=%.2f sec=%.1f kw=%.2f bp=%.2f final=%.2f",
                r.chunk_id, ce_score, section_bonus, keyword_bonus,
                boilerplate_penalty, r.score,
            )

            rerank_hits.append({
                "chunk_id": r.chunk_id,
                "ticker": r.metadata.get("ticker"),
                "section": r.section,
                "ce_score": float(ce_score),
                "section_bonus": section_bonus,
                "keyword_bonus": keyword_bonus,
                "boilerplate_penalty": boilerplate_penalty,
                "final_score": r.score,
            })

        observer.log_reranking(section_hint=section_hint, hits=rerank_hits)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
