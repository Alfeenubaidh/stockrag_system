from __future__ import annotations

import sys
import os
import re
from typing import Any

# Allow imports from sibling generation package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from generation.citation_assembler import (
    extract_citations,
    split_sentences,
    validate_citations,
    _is_not_found_placeholder,
)

# Weights
_W_CITATION_VALID = 0.30
_W_GROUNDING = 0.25
_W_COVERAGE = 0.20
_W_FAITHFULNESS = 0.15
_W_COMPLETENESS = 0.10


def _norm_section(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _build_valid_citation_keys(
    retrieved_chunks: list[dict[str, Any]]
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for chunk in retrieved_chunks:
        ticker = chunk.get("ticker", "").upper()
        section = _norm_section(chunk.get("section", ""))
        if ticker and section:
            keys.add((ticker, section))
    return keys


def _iter_all_sentences(
    answers: dict[str, str], comparison: str
) -> list[str]:
    sentences: list[str] = []

    # Answers
    for text in answers.values():
        if not text or _is_not_found_placeholder(text):
            continue
        for s in split_sentences(text):
            if not _is_not_found_placeholder(s):
                sentences.append(s)

    # Comparison
    if comparison and not _is_not_found_placeholder(comparison):
        for s in split_sentences(comparison):
            if not _is_not_found_placeholder(s):
                sentences.append(s)

    return sentences


def _parse_citation_key(citation: str) -> tuple[str, str] | None:
    inner = citation.strip("()[] ")

    # New format: [AAPL 10-K · FY2024 · Risk Factors] or [AAPL 10-K · Risk Factors]
    if "\u00b7" in inner:
        parts = [p.strip() for p in inner.split("\u00b7")]
        ticker = parts[0].split()[0].upper()  # "AAPL" from "AAPL 10-K"
        section = _norm_section(parts[-1])
        return ticker, section

    # Legacy format: [AAPL, Risk Factors]
    parts = inner.split(",", 1)
    if len(parts) != 2:
        return None
    ticker = parts[0].strip().upper()
    section = _norm_section(parts[1])
    return ticker, section


def score_generation(
    output: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, float]:

    answers_raw: dict[str, str] = output.get("answers", {})
    comparison: str = output.get("comparison", "Not found in filings")

    # Normalize answer keys
    answers = {k.strip().upper(): v for k, v in answers_raw.items()}

    expected_tickers: list[str] = [t.upper() for t in expected.get("tickers", [])]
    expected_keywords: list[str] = [kw.lower() for kw in expected.get("keywords", [])]

    # --- citation validation ---
    validation_result = validate_citations(answers, comparison)
    citation_valid = 1.0 if validation_result.valid else 0.0

    # --- collect sentences ---
    all_sentences = _iter_all_sentences(answers, comparison)

    # --- grounding ---
    valid_keys = _build_valid_citation_keys(retrieved_chunks)

    total_citations = 0
    grounded_citations = 0

    for sentence in all_sentences:
        for cite in extract_citations(sentence):
            total_citations += 1
            key = _parse_citation_key(cite)
            if key and key in valid_keys:
                grounded_citations += 1

    if total_citations == 0:
        grounding = 0.0
        citation_valid = 0.0
    else:
        grounding = grounded_citations / total_citations

    # --- keyword coverage (token-based) ---
    all_text = " ".join(
        text.lower()
        for text in answers.values()
        if text and not _is_not_found_placeholder(text)
    )

    tokens = set(all_text.split())

    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw in tokens)
        coverage = matched / len(expected_keywords)
    else:
        coverage = 1.0

    # --- completeness ---
    if expected_tickers:
        answered = sum(
            1
            for t in expected_tickers
            if t in answers and not _is_not_found_placeholder(answers[t])
        )
        completeness = answered / len(expected_tickers)
    else:
        completeness = 1.0

    # --- faithfulness = grounding proxy ---
    faithfulness = grounding

    # --- final score ---
    score = (
        _W_CITATION_VALID * citation_valid
        + _W_GROUNDING * grounding
        + _W_COVERAGE * coverage
        + _W_FAITHFULNESS * faithfulness
        + _W_COMPLETENESS * completeness
    )

    return {
        "citation_valid": round(citation_valid, 4),
        "grounding": round(grounding, 4),
        "coverage": round(coverage, 4),
        "faithfulness": round(faithfulness, 4),
        "completeness": round(completeness, 4),
        "score": round(score, 4),
    }