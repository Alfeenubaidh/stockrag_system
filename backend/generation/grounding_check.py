from __future__ import annotations

import re
from typing import Any

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were",
    "it", "its", "in", "of", "to", "and", "or", "for", "on", "with",
    "that", "this", "be", "by", "at", "from", "as", "have", "has",
})

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _keywords(text: str) -> frozenset[str]:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return frozenset(w for w in cleaned.split() if w and w not in _STOPWORDS)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _check_sentence(
    sentence: str,
    ticker: str,
    chunks: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    s_words = _keywords(sentence)
    if not s_words:
        return False, None

    ticker_upper = ticker.upper()
    best_overlap = 0
    best_id: str | None = None

    for chunk in chunks:
        if chunk.get("ticker", "").upper() != ticker_upper:
            continue
        c_words = _keywords(chunk.get("text", ""))
        overlap = len(s_words & c_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = chunk.get("chunk_id") or chunk.get("id") or str(chunks.index(chunk))

    grounded = best_overlap >= 3
    return grounded, best_id if grounded else None


def check_grounding(
    answer: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify each sentence in answer["answers"] is supported by at least one
    retrieved chunk. Attaches a "grounding" key to the answer dict.
    """
    grounding: dict[str, Any] = {}

    for ticker, text in answer.get("answers", {}).items():
        if not isinstance(text, str) or text.strip() in ("", "Not found in filings"):
            grounding[ticker] = {
                "sentences": [],
                "grounded_ratio": 1.0,
                "ungrounded_sentences": [],
            }
            continue

        sentences = _split_sentences(text)
        results: list[dict[str, Any]] = []
        ungrounded: list[str] = []

        for sentence in sentences:
            grounded, matched_id = _check_sentence(sentence, ticker, chunks)
            results.append({
                "text": sentence,
                "grounded": grounded,
                "matched_chunk_id": matched_id,
            })
            if not grounded:
                ungrounded.append(sentence)

        total = len(results)
        grounded_count = sum(1 for r in results if r["grounded"])
        ratio = grounded_count / total if total else 1.0

        grounding[ticker] = {
            "sentences": results,
            "grounded_ratio": round(ratio, 3),
            "ungrounded_sentences": ungrounded,
        }

    return {**answer, "grounding": grounding}
