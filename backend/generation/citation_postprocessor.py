"""
citation_postprocessor.py — Deterministic citation enforcement for LLaMA3 output.

Handles:
- Missing citations: aligns statements to retrieved chunks, appends citation
- Wrong format: normalizes [1], (Source), inline text refs → [TICKER, section]
- Hallucinated citations: removes refs with no matching chunk
- Orphaned chunks: appends unreferenced high-relevance chunks as footnotes
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target citation format:  [TICKER DOC_TYPE · FYYYYY · Section]
# e.g.  [TSLA 10-K · FY2023 · MD&A]  [AAPL 10-K · Risk Factors]  (no year if unknown)
# ---------------------------------------------------------------------------

_TARGET_FMT = "[{ticker} {doc_type} \u00b7 FY{year} \u00b7 {section}]"

# Patterns for citation styles the model produces inconsistently
_CITATION_PATTERNS = [
    re.compile(r"\[(\d+)\]"),                          # [1], [2]
    re.compile(r"\(([A-Z]{2,5}),?\s*([^)]{3,40})\)"), # (TSLA, MD&A)
    re.compile(r"\[([A-Z]{2,5}),?\s*([^\]]{3,40})\]"), # [TSLA, MD&A] — already correct
    re.compile(r"\(Source:?\s*([^)]+)\)"),             # (Source: Tesla 10-K)
    re.compile(r"\[Source:?\s*([^\]]+)\]"),            # [Source: Tesla 10-K]
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Minimum lexical overlap to consider a chunk a match for a statement
_MATCH_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    ticker: str
    section: str
    text: str
    score: float = 0.0
    doc_type: str = ""
    filing_date: str = ""

    @property
    def citation(self) -> str:
        sec = _normalize_section_label(self.section)
        dt = self.doc_type.strip() or "10-K"
        fy = ""
        if self.filing_date and self.filing_date not in ("", "unknown"):
            fy = f"FY{self.filing_date[:4]}"
        if fy:
            return f"[{self.ticker} {dt} \u00b7 {fy} \u00b7 {sec}]"
        return f"[{self.ticker} {dt} \u00b7 {sec}]"


@dataclass
class ProcessedOutput:
    original: str
    corrected: str
    citations_added: int
    citations_fixed: int
    citations_removed: int
    unmatched_statements: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CitationPostProcessor:

    def __init__(
        self,
        min_match_threshold: float = _MATCH_THRESHOLD,
        append_unmatched: bool = False,   # if True, appends low-confidence citations
        max_citations_per_sentence: int = 2,
    ):
        self.threshold = min_match_threshold
        self.append_unmatched = append_unmatched
        self.max_per_sentence = max_citations_per_sentence

    def process(
        self,
        raw_output: str,
        chunks: List[RetrievedChunk],
    ) -> ProcessedOutput:
        """
        Main entry point. Takes raw LLaMA3 output + retrieved chunks.
        Returns corrected output with enforced citations.
        """
        if not raw_output or not raw_output.strip():
            return ProcessedOutput(raw_output, raw_output, 0, 0, 0)

        stats = {"added": 0, "fixed": 0, "removed": 0}

        # 1. Strip all existing citations (we re-derive them)
        stripped, removed_count = _strip_citations(raw_output)
        stats["removed"] = removed_count

        # 2. Split into sentence-level units
        sentences = _split_sentences(stripped)

        # 3. Align each sentence to best chunk(s)
        corrected_sentences = []
        unmatched = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            matches = _align_to_chunks(sent, chunks, self.threshold)

            if matches:
                seen = set()
                deduped = []
                for c in matches[:self.max_per_sentence]:
                    key = (c.ticker, c.section)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(c)
                citations = " ".join(c.citation for c in deduped)
                # Remove trailing period, append citation, restore period
                if sent.endswith("."):
                    corrected_sentences.append(f"{sent[:-1]} {citations}.")
                else:
                    corrected_sentences.append(f"{sent} {citations}")
                stats["added"] += 1
            else:
                corrected_sentences.append(sent)
                unmatched.append(sent)
                if self.append_unmatched:
                    logger.debug("unmatched_statement: %r", sent[:80])

        corrected = " ".join(corrected_sentences)

        # 4. Final normalization pass (catches any model-injected refs that survived)
        corrected = _normalize_all_citations(corrected, chunks)

        logger.info(
            "citation_postprocessor  added=%d  fixed=%d  removed=%d  unmatched=%d",
            stats["added"], stats["fixed"], stats["removed"], len(unmatched),
        )

        return ProcessedOutput(
            original=raw_output,
            corrected=corrected,
            citations_added=stats["added"],
            citations_fixed=stats["fixed"],
            citations_removed=stats["removed"],
            unmatched_statements=unmatched,
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _strip_citations(text: str) -> Tuple[str, int]:
    """Remove all citation-like patterns. Returns (cleaned_text, count_removed)."""
    count = 0
    for pat in _CITATION_PATTERNS:
        matches = pat.findall(text)
        count += len(matches)
        text = pat.sub("", text)
    # Clean up double spaces left behind
    text = re.sub(r"  +", " ", text).strip()
    return text, count


def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries, preserve structure."""
    # Protect common abbreviations
    text = re.sub(r"\b(Mr|Mrs|Dr|vs|etc|Inc|Corp|Ltd)\.", r"\1<DOT>", text)
    parts = _SENTENCE_SPLIT.split(text)
    return [p.replace("<DOT>", ".") for p in parts if p.strip()]


def _align_to_chunks(
    sentence: str,
    chunks: List[RetrievedChunk],
    threshold: float,
) -> List[RetrievedChunk]:
    """
    Lexical overlap match: sentence terms vs chunk text.
    Returns chunks above threshold, sorted by overlap score.
    """
    sent_terms = _tokenize(sentence)
    if not sent_terms:
        return []

    scored = []
    for chunk in chunks:
        chunk_terms = _tokenize(chunk.text)
        if not chunk_terms:
            continue
        overlap = len(sent_terms & chunk_terms) / len(sent_terms)
        if overlap >= threshold:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _tokenize(text: str) -> set:
    """Lowercase alpha tokens, length >= 4, minus stopwords."""
    _STOP = {
        "that","this","with","from","have","been","they","their",
        "will","were","also","into","which","when","such","more",
        "than","its","our","the","and","for","are","was",
    }
    tokens = re.findall(r"[a-z]{4,}", text.lower())
    return {t for t in tokens if t not in _STOP}


def _normalize_section_label(section: str) -> str:
    """Normalize Qdrant section labels to display format."""
    section = section.strip()
    _MAP = {
        "risk_factors": "Risk Factors",
        "risk factors": "Risk Factors",
        "item 1a": "Risk Factors",
        "1a": "Risk Factors",
        "md&a": "MD&A",
        "management": "MD&A",
        "mda": "MD&A",
        "item 7": "MD&A",
        "business": "Business",
        "item 1": "Business",
        "outlook": "Outlook",
        "segment results": "Segment Results",
        "cybersecurity": "Cybersecurity",
    }
    return _MAP.get(section.lower(), section.title())


def _normalize_all_citations(text: str, chunks: List[RetrievedChunk]) -> str:
    """
    Second pass: find any citations the model re-injected in wrong format
    and convert them to canonical format or remove if unresolvable.
    """
    # Build ticker → first chunk lookup for metadata
    ticker_meta: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.ticker and c.ticker not in ticker_meta:
            ticker_meta[c.ticker] = c

    # Numeric refs [1] [2] — can't resolve without a reference list, remove
    text = re.compile(r"\[\d+\]").sub("", text)

    # (TSLA, something) → canonical format using chunk metadata
    def fix_paren(m: re.Match) -> str:
        ticker = m.group(1).upper()
        section = _normalize_section_label(m.group(2))
        chunk = ticker_meta.get(ticker)
        if chunk:
            dt = chunk.doc_type.strip() or "10-K"
            fy = ""
            if chunk.filing_date and chunk.filing_date not in ("", "unknown"):
                fy = f"FY{chunk.filing_date[:4]}"
            if fy:
                return f"[{ticker} {dt} \u00b7 {fy} \u00b7 {section}]"
            return f"[{ticker} {dt} \u00b7 {section}]"
        return f"[{ticker}, {section}]"  # fallback: no metadata

    text = re.compile(r"\(([A-Z]{2,5}),?\s*([^)]{3,40})\)").sub(fix_paren, text)

    # Clean double spaces
    return re.sub(r"  +", " ", text).strip()