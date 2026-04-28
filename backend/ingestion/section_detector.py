"""
section_detector.py — assigns section labels to chunks (PRODUCTION VERSION)
"""

import re
import logging
from typing import List, Optional

from ingestion.models import Chunk

logger = logging.getLogger(__name__)


class SectionDetector:

    # ---------------------------------------------------------------
    # SEC Section Mapping — 10-K
    # ---------------------------------------------------------------

    SECTION_MAP_10K = {
        "1":  "Item 1: Business",
        "1A": "Item 1A: Risk Factors",
        "1B": "Item 1B: Unresolved Staff Comments",
        "1C": "Item 1C: Cybersecurity",
        "2":  "Item 2: Properties",
        "3":  "Item 3: Legal Proceedings",
        "7":  "Item 7: Management Discussion and Analysis",
        "7A": "Item 7A: Market Risk Disclosures",
        "8":  "Item 8: Financial Statements",
    }

    # ---------------------------------------------------------------
    # SEC Section Mapping — 10-Q
    # 10-Q Item structure differs from 10-K:
    #   Part I: Item 1 = Financial Statements, Item 2 = MD&A,
    #           Item 3 = Market Risk, Item 4 = Controls
    #   Part II: Item 1 = Legal, Item 1A = Risk Factors, Item 5/6 = Other
    # ---------------------------------------------------------------

    SECTION_MAP_10Q = {
        "1":  "Item 1: Financial Statements",
        "1A": "Item 1A: Risk Factors",
        "2":  "Item 2: Management Discussion and Analysis",
        "3":  "Item 3: Market Risk Disclosures",
        "4":  "Item 4: Controls and Procedures",
        "5":  "Item 5: Other Information",
        "6":  "Item 6: Exhibits",
    }

    # Keep SECTION_MAP as alias for 10-K (backward compat)
    SECTION_MAP = SECTION_MAP_10K

    # ---------------------------------------------------------------
    # Earnings patterns (tightened)
    # ---------------------------------------------------------------

    EARNINGS_PATTERNS = [
        (re.compile(r"\boutlook\b", re.IGNORECASE), "Outlook"),
        (re.compile(r"\bhighlights?\b", re.IGNORECASE), "Business Highlights"),
        (re.compile(r"\bnon[-\s]?gaap\b", re.IGNORECASE), "Non-GAAP"),
        (re.compile(r"\bcash\s+flow\b", re.IGNORECASE), "Cash Flow"),
        (re.compile(r"\bbalance\s+sheet\b", re.IGNORECASE), "Balance Sheet"),
        (re.compile(r"\brevenue\b", re.IGNORECASE), "Revenue"),
        (re.compile(r"\bearnings\b", re.IGNORECASE), "Earnings"),
    ]

    MIN_TEXT_LEN = 80

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def assign(self, chunks: List[Chunk]) -> List[Chunk]:
        current_section: Optional[str] = None
        detected_count = 0

        for chunk in chunks:
            strong = self._detect_strong(chunk)

            if strong:
                current_section = strong
                detected_count += 1
            else:
                content = self._detect_content_based(chunk.text)
                if content:
                    current_section = content
                elif current_section is None:
                    fallback = self._fallback_section(chunk.text.strip())
                    if fallback:
                        current_section = fallback

            chunk.section = current_section

        total = len(chunks)
        ratio = detected_count / total if total else 0
        logger.info(
            f"SectionDetector — detected {detected_count}/{total} "
            f"({ratio:.2%}) explicit section headers"
        )
        return chunks

    # ---------------------------------------------------------------
    # Detection Router
    # ---------------------------------------------------------------

    def _detect_strong(self, chunk: Chunk) -> Optional[str]:
        """Strong signal only — SEC headers and earnings sections. No fallback."""
        text = chunk.text.strip()
        if len(text) < self.MIN_TEXT_LEN:
            return None

        doc_type = (chunk.doc_type or "").upper()

        if "EARNINGS" in doc_type or "TRANSCRIPT" in doc_type:
            return self._detect_earnings(text)

        # Route to correct section map by doc_type
        if "10-Q" in doc_type or "10Q" in doc_type:
            return self._detect_sec(text, self.SECTION_MAP_10Q)

        return self._detect_sec(text, self.SECTION_MAP_10K)

    # Bare section title lines (no "Item X." prefix) — AAPL and similar PDFs
    _BARE_TITLES: dict[str, str] = {
        "risk factors":                     "Risk Factors",
        "legal proceedings":                "Legal Proceedings",
        "properties":                       "Properties",
        "management":                       "MD&A",
        "management's discussion":          "MD&A",
        "quantitative and qualitative":     "Market Risk",
        "financial statements":             "Financial Statements",
        "notes to consolidated":            "Financial Statements",
        "selected financial data":          "Financial Statements",
        "cybersecurity":                    "Cybersecurity",
        "business":                         "Business",
        "exhibits":                         "Exhibits",
    }

    # ---------------------------------------------------------------
    # SEC Detection (strict, structure-aware)
    # ---------------------------------------------------------------

    def _detect_sec(self, text: str, section_map: dict) -> Optional[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Bare title match: first non-empty line is a known section heading
        if lines:
            first = lines[0].lower().rstrip(".")
            for title_key, section_label in self._BARE_TITLES.items():
                if first == title_key or first.startswith(title_key):
                    return section_label

        for i, line in enumerate(lines):

            # Case 1: "ITEM 1"
            match = re.match(
                r"^ITEM\s+(\d{1,2}[A-Z]?)\.?\s*$",
                line,
                re.IGNORECASE,
            )

            if match:
                code = match.group(1).upper()

                if code not in section_map:
                    continue

                # Look ahead for title
                if i + 1 < len(lines):
                    next_line = lines[i + 1]

                    if (
                        len(next_line.split()) <= 12
                        and next_line[0].isupper()
                        and not next_line.endswith(".")
                    ):
                        return f"Item {code}: {next_line.title()}"

                return section_map[code]

            # Case 2: "ITEM 1. Business" (title may be long — no char cap)
            match_inline = re.match(
                r"^ITEM\s+(\d{1,2}[A-Z]?)\.\s+([A-Z][A-Za-z ,&\-']{3,})$",
                line,
                re.IGNORECASE,
            )

            if match_inline:
                code = match_inline.group(1).upper()
                title = match_inline.group(2).strip().title()

                if code in section_map:
                    return f"Item {code}: {title}"

        return None

    # ---------------------------------------------------------------
    # Earnings Detection
    # ---------------------------------------------------------------

    def _detect_earnings(self, text: str) -> Optional[str]:
        hits = []

        for pattern, section in self.EARNINGS_PATTERNS:
            if pattern.search(text):
                hits.append(section)

        if len(hits) >= 2:
            return hits[0]

        return None

    # ---------------------------------------------------------------
    # Content-based classification (no Item headers present)
    # ---------------------------------------------------------------

    SECTION_KEYWORDS: dict[str, list[str]] = {
        "risk_factors": [
            "uncertainty", "adverse", "could harm", "may negatively",
            "could affect", "could result", "could cause", "no assurance",
            "cannot predict", "material adverse", "may not be able",
            "there can be no assurance", "risk factor",
        ],
        "business": [
            "products and services", "seasonality", "product introductions",
            "applecare", "retail stores", "app store",
            "iphone", "ipad", "mac", "apple watch", "apple tv",
            "research and development", "fiscal year ended",
        ],
        "legal_proceedings": [
            "filed a complaint", "filed suit", "district court",
            "class action", "epic games", "digital markets act",
            "plaintiff", "defendant",
        ],
        "md&a": [
            "results of operations", "net sales", "gross margin",
            "operating expenses", "revenue", "fiscal 202", "compared to",
            "increased", "decreased", "management discussion",
        ],
        "financial_statements": [
            "consolidated balance", "consolidated statements", "total assets",
            "shareholders equity", "cash and cash equivalents",
            "notes to consolidated", "fair value", "deferred revenue",
            "interest payments", "deferred tax",
        ],
        "exhibits": [
            "indenture", "rsu", "restricted stock unit", "vesting",
            "termination of service", "nonforfeit", "participant",
            "award agreement", "performance period", "insider trading",
            "trading plan", "blackout period",
        ],
        "properties": [
            "headquarters", "cupertino", "leased facilities", "square feet",
            "owned or leased", "data centers",
        ],
    }

    _SECTION_DISPLAY: dict[str, str] = {
        "risk_factors":        "Risk Factors",
        "business":            "Business",
        "legal_proceedings":   "Legal Proceedings",
        "md&a":                "MD&A",
        "financial_statements": "Financial Statements",
        "exhibits":            "Exhibits",
        "properties":          "Properties",
    }

    _SECTION_MIN_HITS: dict[str, int] = {
        "legal_proceedings": 3,
        "exhibits": 2,
    }

    def _detect_content_based(self, text: str) -> Optional[str]:
        window = text[:300].lower()
        scores: dict[str, int] = {}
        for section, keywords in self.SECTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in window)
            min_hits = self._SECTION_MIN_HITS.get(section, 2)
            if hits >= min_hits:
                scores[section] = hits
        if not scores:
            return None
        best_section = max(scores, key=lambda s: scores[s])
        return self._SECTION_DISPLAY[best_section]

    # ---------------------------------------------------------------
    # Fallback (gap-fill only, never overrides established section)
    # ---------------------------------------------------------------

    def _fallback_section(self, text: str) -> Optional[str]:
        text_lower = text.lower()

        if "risk" in text_lower:
            return "Risk Discussion"

        if "financial" in text_lower:
            return "Financial Discussion"

        if "overview" in text_lower:
            return "Overview"

        return None