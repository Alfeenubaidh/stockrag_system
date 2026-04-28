"""
tests/generation_tests.py — Rigorous generation layer test harness.

Tests generate_answer() end-to-end: retrieval → generation → citation validation.

Requires:
    - Qdrant running at localhost:6333 with sec_filings collection populated
    - ANTHROPIC_API_KEY in environment

Usage:
    python tests/generation_tests.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from embeddings.embedder import EmbeddingConfig, EmbeddingPipeline
from generation.citation_assembler import (
    CITATION_PATTERN,
    _is_not_found_placeholder,
    split_sentences,
)
from generation.generator import generate_answer
from retrieval.query_interface import QueryInterface
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import CrossEncoderReranker
from retrieval.retrieval import RetrievalConfig, RetrievalPipeline
from vector_store.qdrant_client import COLLECTION_NAME, get_qdrant_client

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

_SEP  = "=" * 72
_DASH = "-" * 72

# Sentences longer than this (chars) are flagged as likely split failures
_MAX_SENTENCE_CHARS = 500
# Minimum real words required in a sentence after stripping all citation tokens
_MIN_CONTENT_WORDS = 3
# Tokens excluded from query keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "what", "which", "who", "how", "when",
    "did", "does", "do", "say", "are", "is", "was", "were", "has", "have",
    "be", "been", "will", "would", "could", "should", "may", "might",
    "about", "its", "their", "also", "between", "key",
})
# Minimum ratio (exclusive) of cited-section tokens to retrieved-section tokens
# for a section match. Prevents single-token partial matches:
#   "Risk" (1) vs "Risk Factors" (2) → 1/2 = 0.5, NOT > 0.5 → rejected.
#   "Risk Factors" (2) vs "Risk Factors" (2) → 2/2 = 1.0 > 0.5 → accepted.
_SECTION_MATCH_COVERAGE_THRESHOLD = 0.5
# Known company-name aliases for ticker symbols — excluded from query alignment
# keyword matching to prevent trivial matches (e.g. "apple" always appears in
# AAPL answers and carries no topical signal about the actual query subject).
_TICKER_NAMES: dict[str, frozenset[str]] = {
    "AAPL":  frozenset({"apple"}),
    "MSFT":  frozenset({"microsoft"}),
    "TSLA":  frozenset({"tesla"}),
    "GOOGL": frozenset({"google", "alphabet"}),
    "GOOG":  frozenset({"google", "alphabet"}),
    "AMZN":  frozenset({"amazon"}),
    "META":  frozenset({"meta", "facebook"}),
    "NVDA":  frozenset({"nvidia"}),
    "NFLX":  frozenset({"netflix"}),
}
# Words that indicate genuine comparison language; at least one must appear in
# a non-placeholder comparison field when ≥2 substantive answers are present.
_CONTRAST_INDICATORS = frozenset({
    "whereas", "while", "compared", "however", "unlike", "contrast",
    "conversely", "similarly", "both", "neither", "difference",
    "similar", "differ", "relative", "versus", "vs",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    category: str
    query: str
    tickers: list[str] | None = None
    sections: list[str] | None = None
    # If True: ALL answers must be "not found" placeholders — any substantive
    # content is a hard FAIL (hallucination).
    expect_not_found: bool = False
    # If True: skip retrieval and pass [] to generate_answer directly.
    force_empty: bool = False
    # Human-readable description of what this test expects — printed during run.
    note: str = ""


@dataclass
class ValidationResult:
    passed: bool
    error: str | None = None
    uncited_sentences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    query: str
    category: str
    passed: bool
    error: str | None
    output: dict[str, Any] | None
    chunk_count: int
    uncited_sentences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":  self.query,
            "passed": self.passed,
            "error":  self.error,
            "output": self.output,
        }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

TEST_CASES: list[TestCase] = [
    # --- A: Happy path ---
    TestCase(
        name="A1 - Apple risk factors",
        category="happy_path",
        query="What are the main risk factors Apple faces?",
        tickers=["AAPL"],
        sections=["risk factors"],
        note="Single ticker, section-filtered. Expect cited AAPL answers.",
    ),
    TestCase(
        name="A2 - Microsoft management discussion",
        category="happy_path",
        query="What does Microsoft's management discussion say about revenue growth and operating segments?",
        tickers=["MSFT"],
        sections=["management discussion"],
        note="Single ticker, section-filtered. Expect cited MSFT answers.",
    ),
    # --- B: Missing info — ALL answers must be placeholders ---
    TestCase(
        name="B1 - Apple quantum computing",
        category="missing_info",
        query="What did Apple say about quantum computing risks?",
        tickers=["AAPL"],
        expect_not_found=True,
        note=(
            "Topic absent from filings. PASS requires every answer to be a 'not found' "
            "placeholder. Any substantive content is a FAIL (hallucination)."
        ),
    ),
    # --- C: Partial coverage ---
    TestCase(
        name="C1 - Apple and Tesla supply chain",
        category="partial_coverage",
        query="Compare Apple and Tesla supply chain risks",
        tickers=["AAPL", "TSLA"],
        note=(
            "TSLA may not be in the collection. Expect partial output: AAPL answer cited, "
            "TSLA answer may be 'not found'. No cross-ticker contamination in either answer."
        ),
    ),
    # --- D: Multi-ticker comparison ---
    TestCase(
        name="D1 - Apple vs Microsoft risks",
        category="multi_ticker",
        query="Compare Apple and Microsoft key risk factors",
        tickers=["AAPL", "MSFT"],
        note=(
            "Two tickers. Each ticker's answer must cite only its own sources. "
            "Cross-ticker citations (AAPL answer citing MSFT) are a FAIL. "
            "Comparison section may cite both."
        ),
    ),
    # --- E: Implicit query — no ticker filter ---
    TestCase(
        name="E1 - Autonomous driving risks",
        category="implicit",
        query="What are the risks related to autonomous driving?",
        note=(
            "No ticker filter — retrieval determines which tickers appear in context. "
            "Expected: answers cite only tickers that were actually retrieved; "
            "no ticker hallucination. Every substantive sentence must have a citation. "
            "An empty result is acceptable if no relevant chunks are found."
        ),
    ),
    # --- F: Adversarial ---
    TestCase(
        name="F1 - Apple risks and general market trends",
        category="adversarial",
        query="Explain Apple risks and also general market trends",
        tickers=["AAPL"],
        note=(
            "Query mixes filing-backed content (Apple risks) with uncitable content "
            "(general market trends). The generator must NOT produce sentences about "
            "market trends without citations. Citation validation is the anti-hallucination "
            "check: any uncited sentence is a FAIL."
        ),
    ),
    # --- G: Forced empty retrieval ---
    TestCase(
        name="G1 - Empty retrieval",
        category="empty_retrieval",
        query="What are Apple's risk factors?",
        force_empty=True,
        note=(
            "generate_answer() called with empty chunk list. "
            "Expected: answers == {}, comparison is a 'not found' placeholder. "
            "Any populated answer is a FAIL."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline() -> QueryInterface:
    qdrant   = get_qdrant_client()
    embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=64))
    pipeline = RetrievalPipeline(
        qdrant=qdrant,
        embedder=embedder,
        collection=COLLECTION_NAME,
        config=RetrievalConfig(top_k=5, score_threshold=0.3, use_mmr=True),
        reranker=CrossEncoderReranker(),
        rewriter=QueryRewriter(),
    )
    return QueryInterface(pipeline)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_json_roundtrip(output: dict) -> str | None:
    """
    Verify output is JSON-serializable and survives a round-trip without data loss.

    Returns an error string on failure, None on success.

    Rationale: generate_answer() returns a plain Python dict. If any value contains
    a non-serializable type (datetime, bytes, custom object), it would serialize to
    JSON for a client but produce a different object on re-parse. Catching this here
    prevents silent corruption at the API boundary.
    """
    try:
        serialized = json.dumps(output)
        reparsed   = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        return f"Output is not JSON-serializable: {exc}"
    if reparsed != output:
        return "Output changed after JSON round-trip — non-JSON-safe types present"
    return None


def _check_sentences(text: str, source: str) -> list[str]:
    """
    Split text into sentences and verify each non-placeholder sentence contains
    at least one citation matching CITATION_PATTERN.

    Args:
        text:   The answer or comparison text.
        source: Label for error messages (e.g. "AAPL", "COMPARISON").

    Returns:
        List of uncited sentence strings; empty list if all sentences pass.
    """
    if not text or _is_not_found_placeholder(text):
        return []
    uncited = []
    for sentence in split_sentences(text):
        if _is_not_found_placeholder(sentence):
            continue
        if not CITATION_PATTERN.search(sentence):
            uncited.append(f"[{source}] {sentence}")
    return uncited


def _check_cross_ticker(expected_ticker: str, text: str) -> list[str]:
    """
    Verify every citation inside an answer for `expected_ticker` references only
    that ticker. Citations to foreign tickers in a single-ticker answer indicate
    either a retrieval leak or an LLM hallucination.

    Comparison is intentionally excluded — it legitimately spans multiple tickers.

    Returns:
        List of violation strings; empty list if all citations are correctly scoped.
    """
    if not text or _is_not_found_placeholder(text):
        return []
    violations = []
    for cite in CITATION_PATTERN.findall(text):
        inner = cite[1:-1]  # strip outer parens
        parts = inner.split(",", 1)
        if len(parts) != 2:
            continue
        cited_ticker = parts[0].strip().upper()
        if cited_ticker != expected_ticker.upper():
            violations.append(
                f"Answer for {expected_ticker.upper()} contains foreign citation "
                f"'{cited_ticker}': {cite}"
            )
    return violations


def _check_grounded_tickers(output: dict, retrieved_tickers: set[str]) -> list[str]:
    """
    Verify every ticker referenced in any citation was present in the retrieved chunks.

    Applies to both answers and comparison. This is the primary anti-hallucination
    check for implicit queries (no ticker filter): if the LLM cites a ticker that
    was never retrieved, it invented that reference.

    Args:
        output:            The full generator output dict.
        retrieved_tickers: Set of ticker strings from the retrieved chunk list.

    Returns:
        List of violation strings; empty list if all citations are grounded.
        Returns empty list if retrieved_tickers is empty (force_empty case —
        there is nothing to ground against).
    """
    if not retrieved_tickers:
        return []

    violations = []
    all_texts: list[str] = list(output.get("answers", {}).values())
    all_texts.append(output.get("comparison", ""))

    seen_violations: set[str] = set()  # deduplicate across repeated citations
    for text in all_texts:
        if not text or _is_not_found_placeholder(text):
            continue
        for cite in CITATION_PATTERN.findall(text):
            inner = cite[1:-1]
            parts = inner.split(",", 1)
            if len(parts) != 2:
                continue
            cited_ticker = parts[0].strip().upper()
            if cited_ticker not in retrieved_tickers and cited_ticker not in seen_violations:
                seen_violations.add(cited_ticker)
                violations.append(
                    f"Citation {cite} references ticker '{cited_ticker}' which was not "
                    f"in retrieved chunks (retrieved: {sorted(retrieved_tickers)})"
                )
    return violations


def _build_retrieved_sections(chunks: list[dict]) -> set[tuple[str, str]]:
    """
    Build a set of (ticker_upper, section_lower) pairs from retrieved chunks.
    Used by _check_cited_sections to verify citations against actual content.
    """
    sections: set[tuple[str, str]] = set()
    for chunk in chunks:
        ticker  = chunk.get("ticker", "").strip().upper()
        section = chunk.get("section", "").strip().lower()
        if ticker and section:
            sections.add((ticker, section))
    return sections


def _normalize_section_tokens(section: str) -> frozenset[str]:
    """
    Tokenize a section name into a frozenset of lowercase alphanumeric tokens.
    Strips possessives before splitting to avoid false partial matches.

    Examples:
        "Risk Factors"              → frozenset({"risk", "factors"})
        "MD&A"                      → frozenset({"md", "a"})
        "Management's Discussion"   → frozenset({"management", "discussion"})
        "10-K"                      → frozenset({"10", "k"})
    """
    cleaned = re.sub(r"'s\b|'s\b", "", section.lower())
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    return frozenset(tokens)


def _check_cited_sections(
    text: str,
    source: str,
    retrieved_sections: set[tuple[str, str]],
) -> list[str]:
    """
    Verify each cited (ticker, section) pair corresponds to a section that was
    actually present in retrieved_chunks.

    Section matching uses token-set comparison to prevent partial matches:
        "Risk" (1 token) vs "Risk Factors" (2 tokens):
            subset? yes — coverage = 1/2 = 0.5 — NOT > threshold → rejected ✓
        "Risk Factors" (2) vs "Risk Factors" (2):
            subset? yes — coverage = 2/2 = 1.0 > threshold → accepted ✓

    A match requires:
        1. All cited tokens present in the retrieved section tokens (subset check).
        2. Coverage ratio > _SECTION_MATCH_COVERAGE_THRESHOLD (default 0.5).

    Skipped (returns []) if retrieved_sections is empty.

    Returns:
        List of violation strings; empty list if all cited sections pass.
    """
    if not text or _is_not_found_placeholder(text) or not retrieved_sections:
        return []
    violations = []
    seen: set[tuple[str, str]] = set()
    for cite in CITATION_PATTERN.findall(text):
        inner = cite[1:-1]
        parts = inner.split(",", 1)
        if len(parts) != 2:
            continue
        cited_ticker  = parts[0].strip().upper()
        cited_section = parts[1].strip().lower()
        key = (cited_ticker, cited_section)
        if key in seen:
            continue
        seen.add(key)
        cited_tokens = _normalize_section_tokens(cited_section)
        matched = False
        for rt, rs in retrieved_sections:
            if rt != cited_ticker:
                continue
            rs_tokens = _normalize_section_tokens(rs)
            if not rs_tokens:
                continue
            if (
                cited_tokens <= rs_tokens
                and len(cited_tokens) / len(rs_tokens) > _SECTION_MATCH_COVERAGE_THRESHOLD
            ):
                matched = True
                break
        if not matched:
            violations.append(
                f"[{source}] {cite} — section '{parts[1].strip()}' not found "
                f"in retrieved chunks for {cited_ticker}"
            )
    return violations


def _check_content_substance(text: str, source: str) -> list[str]:
    """
    Verify that no cited sentence is citation-only.

    A sentence such as "(AAPL, Risk Factors)" passes the citation check but carries
    no information. Strip all CITATION_PATTERN matches and require at least
    _MIN_CONTENT_WORDS real words in the remainder.

    Only examines sentences that DO contain a citation — sentences without citations
    are already caught by _check_sentences.

    Returns:
        List of violation strings; empty list if all sentences have real content.
    """
    if not text or _is_not_found_placeholder(text):
        return []
    violations = []
    for sentence in split_sentences(text):
        if _is_not_found_placeholder(sentence):
            continue
        if not CITATION_PATTERN.search(sentence):
            continue  # No citation — already flagged by _check_sentences
        stripped = CITATION_PATTERN.sub("", sentence).strip()
        # Count tokens that contain at least one alphanumeric character
        real_words = [w for w in stripped.split() if re.search(r"[a-zA-Z0-9]", w)]
        if len(real_words) < _MIN_CONTENT_WORDS:
            violations.append(
                f"[{source}] Citation-only sentence — content after removing citations: "
                f"{repr(stripped[:60])!s} | full: {sentence[:100]}"
            )
    return violations


def _check_long_sentences(text: str, source: str) -> tuple[list[str], list[str]]:
    """
    Inspect sentences exceeding _MAX_SENTENCE_CHARS.

    - Warning: sentence is long but has fewer than 2 citations — possible split
      failure, surfaced as a warning because some disclosures contain long prose.
    - Failure: long sentence AND ≥2 citations — multiple claims are being bundled
      under overlapping citations, which is a citation integrity violation.

    Returns:
        (warnings, failures) where each is a list of message strings.
    """
    if not text or _is_not_found_placeholder(text):
        return [], []
    warnings: list[str] = []
    failures: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= _MAX_SENTENCE_CHARS:
            continue
        n_citations = len(CITATION_PATTERN.findall(sentence))
        if n_citations >= 2:
            failures.append(
                f"[{source}] Long sentence ({len(sentence)} chars) with "
                f"{n_citations} citations — multiple claims bundled under "
                f"overlapping citations: {sentence[:80]}..."
            )
        else:
            warnings.append(
                f"[{source}] Overly long sentence ({len(sentence)} chars > "
                f"{_MAX_SENTENCE_CHARS} limit) — possible split failure: "
                f"{sentence[:80]}..."
            )
    return warnings, failures


def _extract_query_keywords(query: str) -> set[str]:
    """
    Extract meaningful tokens from a query, excluding stop words and short tokens.
    Used by _check_query_alignment.
    """
    tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", query).lower().split()
    return {t for t in tokens if t not in _STOP_WORDS and len(t) >= 4}


def _check_query_alignment(
    answers: dict,
    query: str,
    tickers: list[str] | None = None,
) -> str | None:
    """
    Verify at least one non-ticker query keyword appears in the combined
    substantive answer text (case-insensitive substring match).

    Ticker symbols and their company-name aliases (from _TICKER_NAMES) are
    removed from the keyword pool before checking. This prevents the check from
    passing trivially just because "apple" appears in every AAPL answer —
    the keyword must relate to the actual subject of the query.

    Skipped when:
      - No meaningful non-ticker keywords can be extracted from the query.
      - All answers are "not found" placeholders (empty retrieval is valid).

    Returns:
        Error string if no keywords match; None otherwise.
    """
    keywords = _extract_query_keywords(query)
    if not keywords:
        return None

    # Exclude ticker symbols and their company-name aliases
    excluded: set[str] = set()
    if tickers:
        for ticker in tickers:
            excluded.add(ticker.lower())
            excluded.update(_TICKER_NAMES.get(ticker.upper(), frozenset()))
    keywords -= excluded

    if not keywords:
        return None  # All meaningful tokens were ticker names — skip check

    combined = " ".join(
        txt for txt in answers.values()
        if txt and not _is_not_found_placeholder(txt)
    ).lower()

    if not combined:
        return None  # All placeholders — alignment check does not apply

    if not any(kw in combined for kw in keywords):
        return (
            f"No query keyword found in answer text — possible topic drift. "
            f"Query keywords (excl. ticker names): {sorted(keywords)}"
        )
    return None


def adjust_score(text: str, base_score: float, query: str) -> float:
    """
    Improve ranking by:
    - Boosting high-signal keywords (context relevance)
    - Penalizing generic boilerplate text
    - Using multi-signal logic (stronger than naive token match)
    """

    text_lower = text.lower()
    query_lower = query.lower()

    # --- normalize query (fix punctuation issue like "antitrust?")
    query_terms = set(re.findall(r"[a-z0-9]+", query_lower))

    # --- high-signal keywords
    keywords = {"gaming", "antitrust", "activision", "xbox", "regulatory"}

    # count how many important signals appear in BOTH query + text
    match_count = sum(
        1 for k in keywords
        if k in text_lower and k in query_terms
    )

    # --- multi-signal boosting (CRITICAL)
    if match_count >= 2:
        boost = 0.6
    elif match_count == 1:
        boost = 0.2
    else:
        boost = 0.0

    # --- penalize generic boilerplate (VERY IMPORTANT)
    generic_phrases = [
        "may adversely affect",
        "could adversely affect",
        "could impact",
        "may not be able to",
        "could negatively affect",
        "may experience"
    ]

    penalty = 0.6 if any(p in text_lower for p in generic_phrases) else 0.0

    # --- final score (NO CLAMPING)
    final_score = base_score + boost - penalty

    return final_score


def _check_duplicate_sentences(text: str, source: str) -> list[str]:
    """
    Detect repeated sentences within a single answer or comparison field.

    Repetition indicates a generation artifact (stuck output, padding) or the LLM
    restating the same claim multiple times. Comparison is done on normalized text:
    citations stripped, lowercase, whitespace collapsed, leading/trailing punctuation
    removed — so minor formatting differences don't produce false negatives.

    Returns:
        List of violation strings for every duplicate found; empty list if none.
    """
    if not text or _is_not_found_placeholder(text):
        return []
    seen: dict[str, str] = {}  # normalized → first original
    violations: list[str] = []
    for sentence in split_sentences(text):
        if _is_not_found_placeholder(sentence):
            continue
        normalized = CITATION_PATTERN.sub("", sentence).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip().strip(".,;:!?")
        if not normalized:
            continue
        if normalized in seen:
            violations.append(
                f"[{source}] Duplicate sentence detected — "
                f"original: '{seen[normalized][:80]}' | "
                f"duplicate: '{sentence[:80]}'"
            )
        else:
            seen[normalized] = sentence
    return violations


def _check_comparison_contrast(comparison: str) -> str | None:
    """
    Verify a substantive comparison field contains at least one contrast indicator
    from _CONTRAST_INDICATORS (e.g. "whereas", "however", "compared").

    A comparison without contrast language is a description, not a comparison.
    Skipped when comparison is a placeholder.

    Returns:
        Error string if no contrast indicators present; None otherwise.
    """
    if not comparison or _is_not_found_placeholder(comparison):
        return None
    lower = comparison.lower()
    if not any(indicator in lower for indicator in _CONTRAST_INDICATORS):
        return (
            f"Comparison field contains no contrast indicators "
            f"(e.g. {sorted(_CONTRAST_INDICATORS)[:5]}) — "
            f"appears to be a description, not a genuine comparison"
        )
    return None


def _check_comparison_ticker_count(comparison: str, min_tickers: int = 2) -> str | None:
    """
    Verify comparison cites at least `min_tickers` distinct tickers.

    A comparison section that cites only one ticker is a description, not a
    comparison — it indicates either incomplete retrieval coverage or an LLM that
    ignored the multi-ticker scope of the query.

    Skipped when comparison is a placeholder (no-data response is valid).
    Skipped when comparison has no citations at all (already caught by step 7).

    Returns:
        Error string if fewer than min_tickers distinct tickers are cited; None otherwise.
    """
    if not comparison or _is_not_found_placeholder(comparison):
        return None
    cited_tickers: set[str] = set()
    for cite in CITATION_PATTERN.findall(comparison):
        inner = cite[1:-1]
        parts = inner.split(",", 1)
        if len(parts) == 2:
            cited_tickers.add(parts[0].strip().upper())
    if not cited_tickers:
        return None  # No citations — already caught by sentence-level check
    if len(cited_tickers) < min_tickers:
        return (
            f"Comparison cites only {len(cited_tickers)} ticker(s) "
            f"({sorted(cited_tickers)}) — must reference ≥{min_tickers} "
            f"distinct tickers for a valid comparison"
        )
    return None


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(
    output: Any,
    tc: TestCase,
    retrieved_chunks: list[dict],
) -> ValidationResult:
    """
    Validate generator output against test case expectations.

    Checks (in order):
      1.  JSON round-trip: output is JSON-serializable and survives re-parse.
      2.  Schema: output is a dict with 'answers' (dict) and 'comparison' (str).
      3.  No empty strings in answers values or comparison.
      4.  No generator-level error field in the output.
      5.  [Category G] answers must be empty for forced-empty retrieval.
      6.  Sentence-level citation check — every non-placeholder answer sentence
          must contain at least one citation.
      7.  Sentence-level citation check — comparison, same rules as answers.
      8.  Comparison cites ≥2 distinct tickers when ≥2 answers are substantive.
      9.  Comparison contrast check — comparison must contain a contrast indicator
          (e.g. "whereas", "however") when it is substantive.
      10. Cross-ticker check — answer for ticker X must not cite ticker Y.
      11. Grounded-ticker check — every cited ticker must be in retrieved chunks.
      12. Section existence check — every cited (ticker, section) pair must
          correspond to a section present in retrieved_chunks; uses token-set
          matching with coverage threshold to prevent partial matches.
      13. Minimum content check — no sentence may consist of citations alone;
          stripped text must contain ≥ _MIN_CONTENT_WORDS real words.
      14. Long-sentence detection — sentences over _MAX_SENTENCE_CHARS with ≥2
          citations are a FAIL; with fewer citations are a WARNING.
      15. Duplicate sentence detection — repeated sentences (normalized) are a FAIL.
      16. Query alignment — at least one non-ticker query keyword must appear in
          substantive answers (skipped when all answers are placeholders).
      17. [Category B] FAIL if any substantive answer returned (expect_not_found).
    """
    warnings: list[str] = []

    # 1. JSON round-trip
    if not isinstance(output, dict):
        return ValidationResult(False, error=f"Output is not a dict: {type(output)}")
    json_error = _validate_json_roundtrip(output)
    if json_error:
        return ValidationResult(False, error=json_error)

    # 2. Schema
    for key in ("answers", "comparison"):
        if key not in output:
            return ValidationResult(False, error=f"Missing required key: '{key}'")

    answers:    dict = output["answers"]
    comparison: str  = output["comparison"]

    if not isinstance(answers, dict):
        return ValidationResult(False, error=f"'answers' must be dict, got {type(answers)}")
    if not isinstance(comparison, str):
        return ValidationResult(False, error=f"'comparison' must be str, got {type(comparison)}")

    # 3. No empty strings — answers values and comparison
    for ticker, text in answers.items():
        if isinstance(text, str) and not text.strip():
            return ValidationResult(
                False,
                error=f"Empty string answer for ticker '{ticker}' — use a placeholder, not \"\"",
            )
    if not comparison.strip():
        return ValidationResult(
            False,
            error="'comparison' is an empty string — must be a placeholder or substantive text",
        )

    # 4. Generator-level error
    if output.get("error"):
        return ValidationResult(False, error=f"Generator error: {output['error']}")

    # 5. Category G: forced-empty retrieval must produce empty answers
    if tc.force_empty:
        if answers:
            return ValidationResult(
                False,
                error=(
                    f"Expected empty answers for forced-empty retrieval, "
                    f"got: {sorted(answers.keys())}"
                ),
            )
        return ValidationResult(True)

    # 6. Sentence-level citation check — every answer field, independently
    all_uncited: list[str] = []
    for ticker, text in answers.items():
        all_uncited.extend(_check_sentences(text, ticker))

    if all_uncited:
        return ValidationResult(
            False,
            error=f"{len(all_uncited)} answer sentence(s) missing citations",
            uncited_sentences=all_uncited,
        )

    # 7. Sentence-level citation check — comparison field, explicitly
    comparison_uncited = _check_sentences(comparison, "COMPARISON")
    if comparison_uncited:
        return ValidationResult(
            False,
            error=f"{len(comparison_uncited)} comparison sentence(s) missing citations",
            uncited_sentences=comparison_uncited,
        )

    # 8. Comparison must cite ≥2 distinct tickers when ≥2 answers are substantive
    substantive_count = sum(
        1 for txt in answers.values() if not _is_not_found_placeholder(txt)
    )
    if substantive_count >= 2:
        comp_ticker_error = _check_comparison_ticker_count(comparison, min_tickers=2)
        if comp_ticker_error:
            return ValidationResult(False, error=comp_ticker_error)

    # 9. Comparison contrast check — substantive comparison must use contrast language
    contrast_error = _check_comparison_contrast(comparison)
    if contrast_error:
        return ValidationResult(False, error=contrast_error)

    # 10. Cross-ticker check — applies to individual answers, not comparison
    cross_violations: list[str] = []
    for ticker, text in answers.items():
        cross_violations.extend(_check_cross_ticker(ticker, text))

    if cross_violations:
        return ValidationResult(
            False,
            error=f"{len(cross_violations)} cross-ticker citation(s) in answers",
            uncited_sentences=cross_violations,
        )

    # 11. Grounded-ticker check — all cited tickers must come from retrieved chunks
    retrieved_tickers = {
        c.get("ticker", "").upper()
        for c in retrieved_chunks
        if c.get("ticker")
    }
    grounding_violations = _check_grounded_tickers(output, retrieved_tickers)
    if grounding_violations:
        return ValidationResult(
            False,
            error=f"{len(grounding_violations)} citation(s) reference unretrieved ticker(s)",
            uncited_sentences=grounding_violations,
        )

    # 12. Section existence check — every cited (ticker, section) must be in retrieved chunks
    # Uses token-set matching with coverage threshold (prevents "Risk" matching "Risk Factors").
    retrieved_sections = _build_retrieved_sections(retrieved_chunks)
    section_violations: list[str] = []
    for ticker, text in answers.items():
        section_violations.extend(_check_cited_sections(text, ticker, retrieved_sections))
    section_violations.extend(_check_cited_sections(comparison, "COMPARISON", retrieved_sections))

    if section_violations:
        return ValidationResult(
            False,
            error=f"{len(section_violations)} citation(s) reference section not in retrieved chunks",
            uncited_sentences=section_violations,
        )

    # 13. Minimum content check — no citation-only sentences
    substance_violations: list[str] = []
    for ticker, text in answers.items():
        substance_violations.extend(_check_content_substance(text, ticker))
    substance_violations.extend(_check_content_substance(comparison, "COMPARISON"))

    if substance_violations:
        return ValidationResult(
            False,
            error=f"{len(substance_violations)} sentence(s) contain citations but no substantive content",
            uncited_sentences=substance_violations,
        )

    # 14. Long-sentence detection — FAIL if ≥2 citations in a long sentence;
    #     WARNING if long but fewer than 2 citations (possible split failure only).
    long_failures: list[str] = []
    for ticker, text in answers.items():
        w, f = _check_long_sentences(text, ticker)
        warnings.extend(w)
        long_failures.extend(f)
    w, f = _check_long_sentences(comparison, "COMPARISON")
    warnings.extend(w)
    long_failures.extend(f)

    if long_failures:
        return ValidationResult(
            False,
            error=f"{len(long_failures)} long sentence(s) bundle multiple citations",
            uncited_sentences=long_failures,
        )

    # 15. Duplicate sentence detection — repeated sentences are a generation artifact
    dup_violations: list[str] = []
    for ticker, text in answers.items():
        dup_violations.extend(_check_duplicate_sentences(text, ticker))
    dup_violations.extend(_check_duplicate_sentences(comparison, "COMPARISON"))

    if dup_violations:
        return ValidationResult(
            False,
            error=f"{len(dup_violations)} duplicate sentence(s) detected",
            uncited_sentences=dup_violations,
        )

    # 16. Query alignment — at least one non-ticker query keyword must appear in answers
    alignment_error = _check_query_alignment(answers, tc.query, tickers=tc.tickers)
    if alignment_error:
        return ValidationResult(False, error=alignment_error)

    # 17. Category B: hard FAIL if any answer is substantive
    if tc.expect_not_found:
        substantive = [
            ticker for ticker, txt in answers.items()
            if not _is_not_found_placeholder(txt)
        ]
        if substantive:
            return ValidationResult(
                False,
                error=(
                    f"Expected 'not found' placeholder for all tickers but received "
                    f"substantive answers for {substantive} — possible hallucination"
                ),
            )

    return ValidationResult(True, warnings=warnings)


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def run_test(
    query: str,
    *,
    tickers: list[str] | None = None,
    sections: list[str] | None = None,
    expect_not_found: bool = False,
    force_empty: bool = False,
    category: str = "ad_hoc",
    note: str = "",
    pipeline: QueryInterface | None = None,
) -> TestResult:
    """
    Run a single generation test: retrieve → generate → validate.

    Prints:
        - query, category, note, filters
        - retrieved chunk count
        - top-3 chunk preview (ticker, section, score, text snippet)
        - full generated output (JSON)
        - validation result (PASS / FAIL + reason, uncited / violated sentences)

    Args:
        query:            User question.
        tickers:          Optional ticker filter for retrieval.
        sections:         Optional section filter for retrieval.
        expect_not_found: Category B — ALL answers must be "not found" placeholders.
        force_empty:      Category G — skip retrieval, pass [] to generator.
        category:         Label printed in output and summary.
        note:             Expected-behavior description; printed before the test runs.
        pipeline:         Pre-built QueryInterface; built on demand if None.

    Returns:
        TestResult with structured pass/fail details.
    """
    tc = TestCase(
        name=query[:60],
        category=category,
        query=query,
        tickers=tickers,
        sections=sections,
        expect_not_found=expect_not_found,
        force_empty=force_empty,
        note=note,
    )

    print(_SEP)
    print(f"QUERY    : {query}")
    print(f"CATEGORY : {category}")
    if tickers:
        print(f"TICKERS  : {tickers}")
    if sections:
        print(f"SECTIONS : {sections}")
    if note:
        print(f"EXPECT   : {note}")
    print(_DASH)

    # --- Retrieval ---
    chunks: list[dict] = []
    if not force_empty:
        if pipeline is None:
            pipeline = build_pipeline()
        try:
            chunks = pipeline.query(query, tickers=tickers, sections=sections, top_k=5)
        except Exception as exc:
            error = f"Retrieval failed: {exc}"
            print(f"[RETRIEVAL ERROR] {error}")
            print(f"VALIDATION : FAIL")
            print(f"  REASON   : {error}")
            return TestResult(
                query=query, category=category, passed=False,
                error=error, output=None, chunk_count=0,
            )

    print(f"CHUNKS   : {len(chunks)} retrieved")
    if chunks:
        print("TOP 3 PREVIEW:")
        for i, chunk in enumerate(chunks[:3], 1):
            ticker  = chunk.get("ticker", "?")
            section = chunk.get("section", "?")
            score   = chunk.get("score", 0.0)
            snippet = chunk.get("text", "")[:120].replace("\n", " ")
            print(f"  {i}. [{ticker} | {section}] score={score:.4f}")
            print(f"     {snippet}...")
    print(_DASH)

    # --- Generation ---
    try:
        output = generate_answer(query, chunks)
    except Exception as exc:
        error = f"generate_answer raised: {exc}"
        print(f"[GENERATION ERROR] {error}")
        print(f"VALIDATION : FAIL")
        print(f"  REASON   : {error}")
        return TestResult(
            query=query, category=category, passed=False,
            error=error, output=None, chunk_count=len(chunks),
        )

    print("GENERATED OUTPUT:")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(_DASH)

    # --- Validation ---
    vr = validate_output(output, tc, chunks)

    if vr.passed:
        status = "PASS" + (" (with warnings)" if vr.warnings else "")
        print(f"VALIDATION : {status}")
        for w in vr.warnings:
            print(f"  WARNING  : {w}")
    else:
        print(f"VALIDATION : FAIL")
        print(f"  REASON   : {vr.error}")
        for s in vr.uncited_sentences:
            print(f"  VIOLATION: {s}")

    return TestResult(
        query=query,
        category=category,
        passed=vr.passed,
        error=vr.error,
        output=output,
        chunk_count=len(chunks),
        uncited_sentences=vr.uncited_sentences,
        warnings=vr.warnings,
    )


def _run_test_case(tc: TestCase, pipeline: QueryInterface) -> TestResult:
    """Unpack a TestCase and delegate to run_test."""
    return run_test(
        query=tc.query,
        tickers=tc.tickers,
        sections=tc.sections,
        expect_not_found=tc.expect_not_found,
        force_empty=tc.force_empty,
        category=f"{tc.category} | {tc.name}",
        note=tc.note,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Stress runner
# ---------------------------------------------------------------------------

def run_all_tests() -> None:
    """
    Run all TEST_CASES with a single shared pipeline. Print per-test traces
    followed by a summary and detailed failure report.

    Exits with code 0 if all tests passed, 1 if any failed.
    """
    print(_SEP)
    print("GENERATION LAYER TEST SUITE")
    print(f"Test cases : {len(TEST_CASES)}")
    print(_SEP)

    try:
        pipeline = build_pipeline()
    except RuntimeError as exc:
        print(f"[FATAL] Cannot build pipeline: {exc}")
        print("Ensure Qdrant is running and ANTHROPIC_API_KEY is set.")
        sys.exit(1)

    results: list[TestResult] = []
    for tc in TEST_CASES:
        result = _run_test_case(tc, pipeline=pipeline)
        results.append(result)
        print()

    # --- Summary ---
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(_SEP)
    print("SUMMARY")
    print(_DASH)
    print(f"TOTAL  : {total}")
    print(f"PASSED : {passed}")
    print(f"FAILED : {failed}")

    if failed:
        print(_DASH)
        print("FAILURES")
        for r in results:
            if r.passed:
                continue
            print()
            print(f"  CATEGORY : {r.category}")
            print(f"  QUERY    : {r.query}")
            print(f"  REASON   : {r.error}")
            if r.uncited_sentences:
                print(f"  VIOLATIONS ({len(r.uncited_sentences)}):")
                for s in r.uncited_sentences:
                    print(f"    - {s}")
            if r.output:
                print("  RAW OUTPUT:")
                for line in json.dumps(r.output, indent=4, ensure_ascii=False).splitlines():
                    print(f"    {line}")

    print(_SEP)
    sys.exit(0 if failed == 0 else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_tests()
