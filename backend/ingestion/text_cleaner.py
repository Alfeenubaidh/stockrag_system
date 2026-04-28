"""
text_cleaner.py — cleans raw PDF page text.

Removes:
- encoding artifacts
- ligatures
- broken line joins
- SEC-specific noise (TOC, page numbers)
"""

import re
import logging
from dataclasses import dataclass
from typing import Callable, List

from ingestion.models import Page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleaning pass abstraction
# ---------------------------------------------------------------------------

@dataclass
class _CleaningPass:
    name: str
    fn: Callable[[str], str]

    def apply(self, text: str) -> str:
        return self.fn(text)


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def _remove_null_bytes(text: str) -> str:
    return text.replace("\x00", "")


def _fix_ligatures(text: str) -> str:
    ligatures = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl",
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)
    return text


def _fix_smart_quotes(text: str) -> str:
    """
    Replace smart quotes and their common mojibake forms.

    PyMuPDF emits U+FFFD (replacement character) for glyphs it cannot map
    from certain embedded PDF fonts — most commonly curly quotes and em-dashes.
    Map the unambiguous sequences first, then strip any remaining U+FFFD so
    they don't pollute chunk text or embeddings.
    """
    replacements = {
        # Smart double quotes
        "\u201c": '"', "\u201d": '"',
        # Smart single quotes / apostrophes
        "\u2018": "'", "\u2019": "'",
        # En-dash / em-dash
        "\u2013": "-", "\u2014": "-",
        # Ellipsis
        "\u2026": "...",
        # Common two-char mojibake sequences produced by PDFs
        # (replacement char used as a stand-in for opening/closing quote pairs)
        "\ufffd\ufffd": '"',
        # Single replacement char — strip rather than guess
        "\ufffd": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _fix_hyphen_breaks(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _fix_sentence_breaks(text: str) -> str:
    return re.sub(r"(?<=[a-zA-Z0-9])\n(?=[a-zA-Z])", " ", text)


def _remove_toc_lines(text: str) -> str:
    """
    Remove table-of-contents lines:
    e.g. "Item 1A. Risk Factors .......... 23"
    """
    return re.sub(
        r"Item\s+\d+[A-Z]?\.\s+.*?\.{2,}\s*\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _remove_page_headers(text: str) -> str:
    """
    Remove common SEC headers/footers
    """
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\d+\n", "\n", text)  # standalone numbers
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    def __init__(self):
        self._passes: List[_CleaningPass] = [
            _CleaningPass("null_bytes", _remove_null_bytes),
            _CleaningPass("ligatures", _fix_ligatures),
            _CleaningPass("smart_quotes", _fix_smart_quotes),
            _CleaningPass("hyphen_breaks", _fix_hyphen_breaks),
            _CleaningPass("sentence_breaks", _fix_sentence_breaks),
            _CleaningPass("toc_removal", _remove_toc_lines),
            _CleaningPass("headers", _remove_page_headers),
            _CleaningPass("whitespace", _normalize_whitespace),
        ]

    def clean(self, pages: List[Page]) -> List[Page]:
        cleaned_pages: List[Page] = []
        dropped = 0

        for page in pages:
            text = self._apply_passes(page.text, page.page)

            if not text or len(text) < 30:
                dropped += 1
                logger.warning("Dropped page %d after cleaning", page.page)
                continue

            cleaned = Page(
                page=page.page,
                text=text,
                doc_id=page.doc_id,
            )
            # Forward extra attributes set by parsers (section, doc_type)
            for attr in ("section", "doc_type"):
                if hasattr(page, attr):
                    setattr(cleaned, attr, getattr(page, attr))
            cleaned_pages.append(cleaned)

        logger.info(
            "TextCleaner: input=%d kept=%d dropped=%d",
            len(pages), len(cleaned_pages), dropped
        )

        return cleaned_pages

    def _apply_passes(self, text: str, page_num: int) -> str:
        for p in self._passes:
            try:
                text = p.apply(text)
            except Exception as e:
                logger.warning(
                    "Cleaning pass failed: %s (page=%d): %s",
                    p.name, page_num, e
                )
        return text