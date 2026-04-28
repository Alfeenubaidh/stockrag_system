"""
evaluation/eval.py — Retrieval quality metrics for StockRAG.

Pure functions — no retrieval, no I/O, no side effects.

Metrics:
  - hit_at_k          : did any relevant chunk appear in top-k?
  - mean_reciprocal_rank : rank of first relevant chunk
  - coverage          : fraction of expected tickers present in results

Relevance definition:
  - ticker match  : doc_id prefix (before first "_") == expected ticker,
                    case-insensitive
  - section match : expected_section is a substring of result.section,
                    case-insensitive
  Both must hold for a result to be considered relevant.
"""

from __future__ import annotations

from retrieval.retrieval import RetrievalResult

# MD&A has many aliases in query files ("management", "outlook", "non-gaap")
# that must map to the actual stored section label "md&a".
_SECTION_ALIASES: dict[str, str] = {
    "management": "md&a",
    "management discussion": "md&a",
    "results of operations": "md&a",
    "outlook": "md&a",
    "non-gaap": "md&a",
    "segment results": "md&a",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ticker_from_doc_id(doc_id: str) -> str:
    """Extract ticker from doc_id prefix (e.g. 'nvda_2024_10k' → 'nvda')."""
    return doc_id.split("_")[0].lower()


def _section_matches(expected: str, actual: str) -> bool:
    e = _SECTION_ALIASES.get(expected.lower(), expected.lower())
    a = actual.lower()
    return e in a or a in e


def _is_relevant(
    result: RetrievalResult,
    expected_tickers: list[str],
    expected_sections: list[str],
) -> bool:
    ticker_ok = _ticker_from_doc_id(result.doc_id) in {t.lower() for t in expected_tickers}
    section_ok = any(_section_matches(s, result.section) for s in expected_sections)
    return ticker_ok and section_ok


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def hit_at_k(
    results: list[RetrievalResult],
    expected_tickers: list[str],
    expected_sections: list[str] | str,
    k: int,
) -> bool:
    """
    Return True if any result in the top-k is relevant.

    Relevant = correct ticker (via doc_id prefix) AND any section matches.
    """
    if isinstance(expected_sections, str):
        expected_sections = [expected_sections]
    return any(
        _is_relevant(r, expected_tickers, expected_sections)
        for r in results[:k]
    )


def mean_reciprocal_rank(
    results: list[RetrievalResult],
    expected_tickers: list[str],
    expected_sections: list[str] | str,
) -> float:
    """
    Return 1/rank of the first relevant result, or 0.0 if none found.

    Relevant = correct ticker AND any section matches.
    """
    if isinstance(expected_sections, str):
        expected_sections = [expected_sections]
    for rank, result in enumerate(results, start=1):
        if _is_relevant(result, expected_tickers, expected_sections):
            return 1.0 / rank
    return 0.0


def coverage(
    results: list[RetrievalResult],
    expected_tickers: list[str],
) -> float:
    """
    Fraction of expected tickers that appear in the result set.

    Only meaningful for multi-ticker (comparison) queries.
    No section filter — checks whether each ticker is represented at all.

    Returns 0.0 if expected_tickers is empty.
    """
    if not expected_tickers:
        return 0.0
    found_tickers = {_ticker_from_doc_id(r.doc_id) for r in results}
    matched = sum(1 for t in expected_tickers if t.lower() in found_tickers)
    return matched / len(expected_tickers)
