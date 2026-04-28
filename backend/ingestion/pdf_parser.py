"""
pdf_parser.py — PyMuPDF-backed implementation of BaseParser.

Responsibility:
    Extract raw text from a PDF, one Page per page.
    Nothing else — no chunking, no section detection.
"""

import logging
import os
import re

import pymupdf as fitz  # PyMuPDF

from ingestion.base_parser import BaseParser
from ingestion.models import Page

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    Extracts text page-by-page using PyMuPDF.

    Notes
    -----
    - Preserves approximate reading order using block sorting
    - Applies minimal normalization only
    - Filters low-signal pages early
    """

    def __init__(self, min_page_chars: int = 150, min_page_words: int = 30):
        self._min_page_chars = min_page_chars
        self._min_page_words = min_page_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, path: str, doc_id: str) -> list[Page]:
        """
        Open *path*, iterate pages, return non-empty Page objects.
        """
        self._validate_path(path)

        try:
            doc = fitz.open(path)
        except Exception as exc:
            raise ValueError(f"Failed to open PDF {path!r}: {exc}") from exc

        pages: list[Page] = []

        with doc:
            if doc.page_count == 0:
                raise ValueError(f"PDF has no pages: {path!r}")

            for page_index in range(doc.page_count):
                page_num = page_index + 1

                text = self._extract_page_text(doc[page_index], page_num)
                if text is None:
                    continue

                pages.append(Page(page=page_num, text=text, doc_id=doc_id))

        if not pages:
            raise ValueError(
                f"PDF yielded no extractable text (all pages empty?): {path!r}"
            )

        logger.info(
            "PDFParser: extracted %d non-empty pages from %r (doc_id=%r)",
            len(pages),
            path,
            doc_id,
        )

        return pages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_path(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path!r}")
        if not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path!r}")

    def _extract_page_text(self, page: fitz.Page, page_num: int) -> str | None:
        """
        Extract text using block layout with approximate reading order.

        Returns None if page is empty or low quality.
        """
        try:
            blocks = page.get_text("blocks")
        except Exception as exc:
            logger.warning("Page %d: block extraction failed — %s", page_num, exc)
            return None

        if not blocks:
            logger.debug("Page %d skipped: no blocks found", page_num)
            return None

        # Sort blocks: top-to-bottom, then left-to-right
        # (rounded y reduces jitter from float noise)
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

        text_parts = []
        for block in blocks:
            if len(block) <= 6:
                continue

            block_type = block[6]
            text = block[4]

            if block_type != 0:
                continue  # skip non-text blocks

            if not text or not text.strip():
                continue

            text_parts.append(text)

        if not text_parts:
            logger.debug("Page %d skipped: no valid text blocks", page_num)
            return None

        raw = "\n".join(text_parts)
        raw = self._normalize_text(raw)

        char_len = len(raw)
        word_len = len(raw.split())

        if char_len < self._min_page_chars or word_len < self._min_page_words:
            logger.debug(
                "Page %d skipped: low content (chars=%d, words=%d)",
                page_num,
                char_len,
                word_len,
            )
            return None

        return raw

    def _normalize_text(self, text: str) -> str:
        """
        Minimal normalization:
        - Replace non-breaking spaces
        - Collapse multiple spaces/tabs
        - Limit excessive newlines
        """
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()