"""
query_rewriter.py — Thin wrapper kept for backward compatibility.

Query expansion is now handled by QueryParser.expanded_queries.
RetrievalPipeline reads expanded_queries directly from ParsedQuery,
so this class is no longer in the hot path.

Kept to avoid breaking any external code that imports QueryRewriter.
"""

from __future__ import annotations

from typing import List

from retrieval.query_parser import QueryParser

_parser = QueryParser()


class QueryRewriter:
    """Delegates to QueryParser.expand for backward compatibility."""

    def rewrite(self, query: str) -> List[str]:
        pq = _parser.parse(query)
        return pq.expanded_queries or [query]
