"""
reranker.py — flashrank reranking with section bonus and ranking signals.

Final scoring formula per chunk:
    final = ce_score
          + section_bonus        (if section_hint matches result.section)
          + keyword_bonus        (informative query term overlap + co-occurrence)
          - boilerplate_penalty  (fraction of text that is generic SEC filler)

Weights are additive so each component acts as a tiebreaker at its scale:
  - CE score range (flashrank ms-marco-MiniLM-L-12-v2): roughly [0, 1]
  - section_bonus: 6.0 (overrides same-section CE variance)
  - keyword_bonus: up to ~3.5 (overrides generic vs. specific chunk gap)
  - boilerplate_penalty: up to 1.5 (demotes filler-heavy chunks)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from flashrank import Ranker, RerankRequest

from retrieval.ranking_signals import RankingSignalScorer
from observability.pipeline_observer import observer

logger = logging.getLogger(__name__)

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
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.model = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        results: List,
        top_k: int,
        section_hint: Optional[str] = None,
    ) -> List:
        """
        Rerank results using flashrank + section bonus + keyword signals.

        Args:
            query:        Original user query (full, untruncated).
            results:      List of RetrievalResult objects.
            top_k:        Number of results to return.
            section_hint: Inferred section string (e.g. "risk factors", "gaming").
        """
        if not results:
            return results

        passages = [{"id": i, "text": r.text} for i, r in enumerate(results)]
        request = RerankRequest(query=query, passages=passages)
        ranked = self.model.rerank(request)

        # ranked is ordered best-first; map scores back by passage id
        score_by_idx = {item["id"]: item["score"] for item in ranked}

        rerank_hits: list[dict] = []
        for i, r in enumerate(results):
            ce_score = score_by_idx.get(i, 0.0)

            section_bonus = _SECTION_BONUS if _section_matches(section_hint, r.section) else 0.0
            keyword_bonus, boilerplate_penalty = _signal_scorer.score(query, r.text)

            r.score = float(ce_score) + section_bonus + keyword_bonus - boilerplate_penalty

            logger.debug(
                "chunk=%s ce=%.4f sec=%.1f kw=%.2f bp=%.2f final=%.4f",
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
