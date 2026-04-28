"""
retrieval/query_interface.py — Thin orchestration layer over RetrievalPipeline.
"""

from typing import Optional, List
from retrieval.retrieval import RetrievalPipeline, RetrievalResult


class QueryInterface:
    def __init__(self, pipeline: RetrievalPipeline):
        self.pipeline = pipeline

    def query(
        self,
        text: str,
        tickers: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        doc_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[dict]:
        results: List[RetrievalResult] = self.pipeline.retrieve(
            query=text,
            doc_type=doc_type,
            sections=sections,
            tickers=tickers,
            top_k=top_k,
        )

        return [
            {
                "rank":     rank,
                "ticker":   r.metadata["ticker"],
                "section":  r.section,
                "score":    round(r.score, 4),
                "text":     r.text,
                "fallback": r.metadata.get("fallback", False),
            }
            for rank, r in enumerate(results, start=1)
        ]