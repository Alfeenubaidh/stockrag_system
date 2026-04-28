from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHUNKS = 5
MAX_CHUNK_TEXT_CHARS = 800
_MAX_PER_TICKER = 3


def build_context(retrieved_chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """
    Build a deduplicated, diversity-preferring context grouped by ticker.

    Args:
        retrieved_chunks: List of retrieval results, each expected to contain:
            - ticker (str)
            - section (str)
            - text (str)
            - chunk_id (str | None, optional)
            - score (float, optional)

    Returns:
        {
            "AAPL": [{"section": "...", "text": "..."}, ...],
            ...
        }
        Total chunks across all tickers <= MAX_CHUNKS.

    Raises:
        ValueError: If retrieved_chunks is not a list.
    """
    if not isinstance(retrieved_chunks, list):
        raise ValueError(f"retrieved_chunks must be a list, got {type(retrieved_chunks)}")

    if not retrieved_chunks:
        return {}

    # --- Deduplication ---
    # Primary key: chunk_id if present; fallback to (ticker, section, text[:120])
    seen: set[str] = set()
    unique_chunks: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        ticker = chunk.get("ticker")
        text = chunk.get("text")
        if not ticker or not text:
            logger.warning(
                "Skipping chunk with missing required fields: ticker=%r text_present=%s",
                ticker,
                bool(text),
            )
            continue
        text_hash = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        dedup_key = (
            chunk["chunk_id"]
            if chunk.get("chunk_id")
            else f"{ticker}|{chunk.get('section', '')}|{text_hash}"
        )
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_chunks.append(chunk)

    # --- Diversity selection (prefer different sections per ticker) ---
    # Sort by score descending so highest-quality chunks win ties
    unique_chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    selected: list[dict[str, Any]] = []
    seen_ticker_sections: set[tuple[str, str]] = set()
    per_ticker_count: dict[str, int] = defaultdict(int)

    # Pass 1: one chunk per (ticker, section) pair — maximises diversity
    for chunk in unique_chunks:
        if len(selected) >= MAX_CHUNKS:
            break
        ticker = chunk.get("ticker", "UNKNOWN")
        section = chunk.get("section", "")
        ts_key = (ticker, section)
        if ts_key not in seen_ticker_sections and per_ticker_count[ticker] < _MAX_PER_TICKER:
            selected.append(chunk)
            seen_ticker_sections.add(ts_key)
            per_ticker_count[ticker] += 1

    # Pass 2: fill remaining slots with highest-scoring leftovers
    if len(selected) < MAX_CHUNKS:
        selected_ids = {id(c) for c in selected}
        for chunk in unique_chunks:
            if len(selected) >= MAX_CHUNKS:
                break
            ticker = chunk.get("ticker", "UNKNOWN")
            if id(chunk) not in selected_ids and per_ticker_count[ticker] < _MAX_PER_TICKER:
                selected.append(chunk)
                per_ticker_count[ticker] += 1

    # --- Group by ticker ---
    context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for chunk in selected:
        ticker = chunk.get("ticker", "UNKNOWN")
        text = chunk.get("text", "")
        context[ticker].append(
            {
                "section": chunk.get("section", ""),
                "text": text[:MAX_CHUNK_TEXT_CHARS],
            }
        )

    return dict(context)