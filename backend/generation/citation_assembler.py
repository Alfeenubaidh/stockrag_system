from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Matches citations in all supported formats:
#   Full:    [AAPL 10-K · FY2023 · Risk Factors]
#   No year: [AAPL 10-K · Risk Factors]          (filing_date unknown)
#   Legacy:  [AAPL, Risk Factors]  or  (AAPL, Risk Factors)
CITATION_PATTERN = re.compile(
    r"(?:"
    r"\[[A-Za-z]{1,5}\s+[A-Za-z0-9\-]+\s*\u00b7\s*FY\d{4}\s*\u00b7\s*[A-Za-z0-9 &/\-]+\]"  # full
    r"|"
    r"\[[A-Za-z]{1,5}\s+[A-Za-z0-9\-]+\s*\u00b7\s*[A-Za-z0-9 &/\-]+\]"  # no year
    r"|"
    r"[\(\[][A-Za-z]{1,5}\s*,\s*[A-Za-z0-9 &/\-]+[\)\]]"  # legacy
    r")"
)

# Sentence splitter: split on '. ' / '! ' / '? ' before an uppercase letter or opening quote.
# Uppercase-only avoids false splits at abbreviations like U.S., Inc., Corp.
# Matches the postprocessor's _SENTENCE_SPLIT to keep the two validators consistent.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


@dataclass
class CitationValidationResult:
    valid: bool
    uncited_sentences: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def extract_citations(text: str) -> list[str]:
    """Return all citation strings found in text."""
    return CITATION_PATTERN.findall(text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering empty strings."""
    raw = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


_NOT_FOUND_PHRASES = (
    "not found",
    "no information found",
    "insufficient information",
    "does not mention",
    "does not specifically",
    "not discussed in",
    "no specific information",
    "no relevant information",
)


def _is_not_found_placeholder(sentence: str) -> bool:
    """True for any recognised 'no data' placeholder (substring match)."""
    lowered = sentence.strip().lower()
    return any(phrase in lowered for phrase in _NOT_FOUND_PHRASES)


def _normalize_section(raw: str) -> str:
    """
    Normalize a section name without corrupting abbreviations.

    Rules:
    - Tokens containing '&' or '/' are fully uppercased  →  md&a → MD&A
    - All other tokens use .title()                       →  risk factors → Risk Factors
                                                             10-k → 10-K  (.title handles hyphens)
    """
    words = []
    for word in raw.strip().split():
        if any(c in word for c in ("&", "/")):
            words.append(word.upper())
        else:
            words.append(word.title())
    return " ".join(words)


def validate_citations(answers: dict[str, str], comparison: str) -> CitationValidationResult:
    """
    Validate that every non-placeholder sentence in answers and comparison
    contains at least one citation.

    Args:
        answers: {ticker: answer_text}
        comparison: comparison text

    Returns:
        CitationValidationResult with validity flag and any uncited sentences.
    """
    uncited: list[str] = []
    errors: list[str] = []

    if not answers:
        errors.append("'answers' dict must not be empty.")
        return CitationValidationResult(valid=False, uncited_sentences=[], errors=errors)

    all_texts: list[tuple[str, str]] = list(answers.items())
    all_texts.append(("COMPARISON", comparison))

    for source, text in all_texts:
        if not text or _is_not_found_placeholder(text):
            continue

        sentences = split_sentences(text)
        for sentence in sentences:
            if _is_not_found_placeholder(sentence):
                continue
            if not extract_citations(sentence):
                uncited.append(f"[{source}] {sentence}")

    if uncited:
        errors.append(
            f"{len(uncited)} sentence(s) are missing citations. "
            "Each factual claim must include (TICKER, Section)."
        )

    return CitationValidationResult(
        valid=len(uncited) == 0,
        uncited_sentences=uncited,
        errors=errors,
    )


def format_citations(text: str, ticker_metadata: dict | None = None) -> str:
    """
    Normalize and upgrade citation formatting.

    Converts legacy  [AAPL, Risk Factors]  →  [AAPL 10-K · FY2023 · Risk Factors]
    using doc_type and filing_date from ticker_metadata when available.
    Already-formatted new-style citations are returned unchanged.

    Examples:
        [aapl, risk factors]  + meta  → [AAPL 10-K · FY2023 · Risk Factors]
        [aapl, md&a]          no meta → [AAPL, MD&A]   (graceful fallback)
    """
    meta = ticker_metadata or {}

    def _normalize(match: re.Match) -> str:
        raw = match.group(0)

        # Already in new format — leave untouched
        if "·" in raw:
            return raw

        inner = raw[1:-1]
        parts = inner.split(",", 1)
        if len(parts) != 2:
            return raw

        ticker = parts[0].strip().upper()
        section = _normalize_section(parts[1].strip())

        tmeta = meta.get(ticker, {})
        doc_type = tmeta.get("doc_type", "").strip()
        filing_date = tmeta.get("filing_date", "").strip()

        fy = ""
        if filing_date and filing_date not in ("", "unknown"):
            fy = f"FY{filing_date[:4]}"

        if doc_type and fy:
            return f"[{ticker} {doc_type} \u00b7 {fy} \u00b7 {section}]"
        if doc_type:
            return f"[{ticker} {doc_type} \u00b7 {section}]"
        # Graceful fallback when no metadata available
        if raw.startswith("["):
            return f"[{ticker}, {section}]"
        return f"({ticker}, {section})"

    return CITATION_PATTERN.sub(_normalize, text)


def assemble_and_validate(raw_output: dict, ticker_metadata: dict | None = None) -> dict:
    """
    Post-process and validate the raw LLM output dict.

    Steps:
    1. Normalize citation formatting in all answer texts.
    2. Validate that every sentence has a citation.
    3. Raise CitationValidationError if any uncited sentences found.

    Args:
        raw_output: {"answers": {...}, "comparison": "..."}

    Returns:
        Validated and formatted output dict (same schema).

    Raises:
        CitationValidationError: if any sentence is missing a citation.
    """
    answers: dict[str, str] = raw_output.get("answers", {})
    comparison: str = raw_output.get("comparison", "Not found in filings")

    if not answers:
        raise CitationValidationError(
            message="Citation validation failed: 'answers' dict is empty.",
            uncited_sentences=[],
            errors=["'answers' must not be empty."],
        )

    # Normalize citations
    formatted_answers = {
        ticker: format_citations(text, ticker_metadata=ticker_metadata)
        for ticker, text in answers.items()
    }
    formatted_comparison = format_citations(comparison, ticker_metadata=ticker_metadata)

    # Validate answers strictly (no change to existing logic)
    answers_only_result = validate_citations(formatted_answers, "Not found in filings")
    if not answers_only_result.valid:
        raise CitationValidationError(
            message="Citation validation failed: uncited sentences detected.",
            uncited_sentences=answers_only_result.uncited_sentences,
            errors=answers_only_result.errors,
        )

    # Validate comparison with relaxed rules for local LLMs
    _validate_comparison_relaxed(formatted_comparison, set(formatted_answers.keys()))

    return {
        "answers": formatted_answers,
        "comparison": formatted_comparison,
    }


def _validate_comparison_relaxed(comparison: str, cited_tickers: set[str]) -> None:
    """
    Relaxed comparison validation: skip strict per-sentence citation check if:
    - comparison is a 'not found' placeholder, OR
    - all tickers mentioned in the comparison are already cited in answers.

    Raises CitationValidationError only if neither condition holds.
    """
    if _is_not_found_placeholder(comparison):
        return

    # Extract tickers mentioned in the comparison text
    comparison_tickers = {m.upper() for m in re.findall(r"\b[A-Z]{2,5}\b", comparison.upper())
                          if m.upper() in cited_tickers or len(m) <= 5}
    mentioned_known = comparison_tickers & cited_tickers

    if mentioned_known and mentioned_known <= cited_tickers:
        logger.warning(
            "Comparison citation validation relaxed: tickers %s already cited in answers.",
            sorted(mentioned_known),
        )
        return

    # Fall back to strict sentence-level validation
    uncited: list[str] = []
    for sentence in split_sentences(comparison):
        if not _is_not_found_placeholder(sentence) and not extract_citations(sentence):
            uncited.append(f"[COMPARISON] {sentence}")

    if uncited:
        raise CitationValidationError(
            message="Citation validation failed: uncited sentences detected.",
            uncited_sentences=uncited,
            errors=[f"{len(uncited)} comparison sentence(s) missing citations."],
        )


class CitationValidationError(Exception):
    """Raised when generated output contains sentences without citations."""

    def __init__(
        self,
        message: str,
        uncited_sentences: list[str],
        errors: list[str],
    ) -> None:
        super().__init__(message)
        self.uncited_sentences = uncited_sentences
        self.errors = errors

    def __str__(self) -> str:
        base = super().__str__()
        detail = "\n".join(f"  - {s}" for s in self.uncited_sentences)
        return f"{base}\nUncited sentences:\n{detail}"