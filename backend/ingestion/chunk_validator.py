"""
chunk_validator.py — ChunkValidator: filters noisy or useless chunks.

Tuned for:
- SEC filings (10-K, 10-Q)
- Earnings transcripts

Filters applied:
    1. MinLengthFilter     — drop very short chunks
    2. NoiseFilter         — drop symbol-heavy chunks
    3. NumericDensityFilter — drop table-like numeric chunks
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from ingestion.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter protocol
# ---------------------------------------------------------------------------

class ChunkFilter(Protocol):
    name: str

    def should_discard(self, chunk: Chunk) -> bool:
        ...


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

@dataclass
class MinLengthFilter:
    name: str = "min_length"
    min_words: int = 30

    def should_discard(self, chunk: Chunk) -> bool:
        return len(chunk.text.split()) < self.min_words


@dataclass
class NoiseFilter:
    name: str = "noise"
    max_noise_ratio: float = 0.4

    _alpha_re: re.Pattern = field(
        default_factory=lambda: re.compile(r"[A-Za-z]{2,}"),
        repr=False,
    )

    def should_discard(self, chunk: Chunk) -> bool:
        tokens = chunk.text.split()

        if not tokens:
            return True

        alpha_count = sum(1 for t in tokens if self._alpha_re.search(t))
        noise_ratio = 1.0 - (alpha_count / len(tokens))

        return noise_ratio > self.max_noise_ratio


@dataclass
class NumericDensityFilter:
    name: str = "numeric_density"
    max_numeric_ratio: float = 0.4

    # Matches:
    # 100, 100.5, 100%, $100M, 1,000, etc.
    _numeric_re: re.Pattern = field(
        default_factory=lambda: re.compile(r"[\d\.,%$]+"),
        repr=False,
    )

    def should_discard(self, chunk: Chunk) -> bool:
        tokens = chunk.text.split()

        if not tokens:
            return True

        numeric_count = sum(1 for t in tokens if self._numeric_re.search(t))
        numeric_ratio = numeric_count / len(tokens)

        return numeric_ratio > self.max_numeric_ratio


# ---------------------------------------------------------------------------
# ChunkValidator
# ---------------------------------------------------------------------------

class ChunkValidator:
    """
    Applies filters to a Chunk list and removes low-quality chunks.
    """

    def __init__(self, filters: list | None = None):
        self._filters: list = filters if filters is not None else [
            MinLengthFilter(),
            NoiseFilter(),
            NumericDensityFilter(),  # ✅ NEW
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, chunks: list[Chunk]) -> list[Chunk]:
        kept: list[Chunk] = []
        discard_counts: dict[str, int] = {f.name: 0 for f in self._filters}

        for chunk in chunks:
            reason = self._first_discard_reason(chunk)

            if reason:
                discard_counts[reason] += 1
            else:
                kept.append(chunk)

        total = len(chunks)
        kept_count = len(kept)
        dropped = total - kept_count

        if dropped > 0:
            logger.info(
                "ChunkValidator — kept %d/%d (%.2f%%). Dropped by filter: %s",
                kept_count,
                total,
                (kept_count / total * 100) if total else 0,
                discard_counts,
            )
        else:
            logger.info("ChunkValidator — all %d chunks passed.", total)

        return kept

    def add_filter(self, f) -> None:
        self._filters.append(f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _first_discard_reason(self, chunk: Chunk) -> str | None:
        for f in self._filters:
            if f.should_discard(chunk):
                return f.name
        return None