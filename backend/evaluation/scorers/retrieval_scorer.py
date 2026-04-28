from __future__ import annotations

import re
from typing import Any

# Weights for composite retrieval score
_W_HIT = 0.35
_W_MRR = 0.25
_W_COVERAGE = 0.20
_W_PRECISION = 0.20


def _norm(text: str) -> str:
    """
    Normalize text for matching:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_retrieval(
    results: list[Any],
    expected: dict[str, list[str]],
    k: int = 3,
) -> dict[str, float]:
    """
    Score retrieval quality against expected tickers and sections.

    Args:
        results: List of RetrievalResult objects or dicts
        expected: {
            "tickers": ["AAPL", "MSFT"],
            "sections": ["Risk Factors", "Management Discussion"]
        }
        k: evaluation cutoff

    Returns:
        dict of retrieval metrics
    """

    expected_tickers: set[str] = {t.upper() for t in expected.get("tickers", [])}
    expected_sections: set[str] = {_norm(s) for s in expected.get("sections", [])}

    # --- guard: invalid expected ---
    if not expected_tickers or not expected_sections:
        return {
            "hit@k": 0.0,
            "mrr": 0.0,
            "coverage": 0.0,
            "precision": 0.0,
            "score": 0.0,
        }

    def _get(result: Any, field: str) -> str:
        if isinstance(result, dict):
            return result.get(field, "")
        return getattr(result, field, "")

    def _section_match(section: str) -> bool:
        section_norm = _norm(section)
        return any(exp in section_norm for exp in expected_sections)

    def _is_relevant(r: Any) -> bool:
        ticker = _get(r, "ticker").upper()
        section = _get(r, "section")
        return ticker in expected_tickers and _section_match(section)

    # --- top-k slice ---
    top_k = results[:k]

    # --- hit@k ---
    hit_at_k = 1.0 if any(_is_relevant(r) for r in top_k) else 0.0

    # --- MRR ---
    mrr = 0.0
    for rank, result in enumerate(top_k, start=1):
        if _is_relevant(result):
            mrr = 1.0 / rank
            break

    # --- strict coverage (ticker + section match) ---
    covered_tickers = set()
    for r in top_k:
        ticker = _get(r, "ticker").upper()
        section = _get(r, "section")
        if ticker in expected_tickers and _section_match(section):
            covered_tickers.add(ticker)

    coverage = len(covered_tickers) / len(expected_tickers)

    # --- precision ---
    relevant_count = sum(1 for r in top_k if _is_relevant(r))
    precision = relevant_count / len(top_k) if top_k else 0.0

    # --- composite score ---
    score = (
        _W_HIT * hit_at_k
        + _W_MRR * mrr
        + _W_COVERAGE * coverage
        + _W_PRECISION * precision
    )

    return {
        "hit@k": round(hit_at_k, 4),
        "mrr": round(mrr, 4),
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "score": round(score, 4),
    }