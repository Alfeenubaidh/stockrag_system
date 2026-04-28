"""
doc_metadata.py

Extracts and normalizes document-level metadata from raw text.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    ticker: str
    doc_type: str
    filing_date: str
    accession_number: str


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MetadataExtractor:
    def extract(self, text: str, filename: str) -> DocumentMetadata:
        ticker = self._extract_ticker(filename)
        doc_type = self._extract_doc_type(filename)
        filing_date = self._extract_filing_date(text)
        accession_number = self._extract_accession_number(text, filename=filename)

        return DocumentMetadata(
            ticker=ticker,
            doc_type=doc_type,
            filing_date=filing_date,
            accession_number=accession_number,
        )

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    def _extract_ticker(self, filename: str) -> str:
        """
        Assumes filename like: AAPL_2024_10K.pdf
        """
        return filename.split("_")[0].upper()

    # ------------------------------------------------------------------
    # Doc Type
    # ------------------------------------------------------------------

    def _extract_doc_type(self, filename: str) -> str:
        """
        Normalize doc_type into canonical forms:
        - 10-K
        - 10-Q
        - earnings_transcript
        """
        name = filename.upper()

        if "10K" in name or "10-K" in name:
            return "10-K"

        if "10Q" in name or "10-Q" in name:
            return "10-Q"

        if "EARNINGS" in name or "TRANSCRIPT" in name:
            return "earnings_transcript"

        return "unknown"

    # ------------------------------------------------------------------
    # Filing Date
    # ------------------------------------------------------------------

    def _extract_filing_date(self, text: str) -> str:
        """
        Extracts date like:
        "For the fiscal year ended September 28, 2024"
        → 2024-09-28
        """

        match = re.search(
            r"fiscal year ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            text,
            re.IGNORECASE,
        )

        if not match:
            return "unknown"

        raw_date = match.group(1)

        return self._normalize_date(raw_date)

    def _normalize_date(self, date_str: str) -> str:
        """
        Converts "September 28, 2024" → "2024-09-28"
        """
        from datetime import datetime

        try:
            dt = datetime.strptime(date_str, "%B %d, %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Accession Number
    # ------------------------------------------------------------------

    def _extract_accession_number(self, text: str, filename: str = "") -> str:
        """
        Extract SEC accession number (format: 0000320193-24-000123).

        Priority:
          1. Filename stem — authoritative when PDF was downloaded from EDGAR
             and renamed to include the accession number.
          2. PDF text body — fallback for filings that embed it on the cover page.
          3. "unknown" — only when neither source yields a match.
        """
        # 1. Filename (most reliable — set at download time).
        #    No \b anchors: underscores are word chars in Python regex, so \b
        #    won't fire between "_" and a digit (e.g. "_0000320193").
        if filename:
            m = re.search(r"\d{10}-\d{2}-\d{6}", Path(filename).stem)
            if m:
                return m.group(0)

        # 2. PDF text body — use \b to avoid matching sub-sequences in numbers.
        m = re.search(r"\b\d{10}-\d{2}-\d{6}\b", text)
        if m:
            return m.group(0)

        return "unknown"