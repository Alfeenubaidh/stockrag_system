import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# SAFE PARSER LOADER
# ------------------------------------------------------------------

def _get_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        logger.warning("lxml not available, falling back to html.parser")
        return BeautifulSoup(html, "html.parser")


# ------------------------------------------------------------------
# XBRL namespaces
# ------------------------------------------------------------------

XBRL_NAMESPACES = {
    "ix", "xbrli", "xbrldi", "xbrldt",
    "link", "dei", "us-gaap", "nonnum",
    "num", "ref", "xs", "xsi",
}


# ------------------------------------------------------------------
# SEC Item detection (FIXED)
# ------------------------------------------------------------------

# Matches SEC Item headers only at line start, optionally followed by a period/dot
# and a title of any length up to 120 chars (covers long titles like Item 7 MD&A).
# Rejects mid-sentence occurrences like "See Item 9A" because those are not at line start.
ITEM_PATTERN = re.compile(
    r"(?m)^ITEM\s+(\d{1,2}[A-Z]?)[\.\s][^\n]{0,120}$",
    re.IGNORECASE,
)


SEC_SECTIONS = {
    "1": "Item 1: Business",
    "1A": "Item 1A: Risk Factors",
    "1B": "Item 1B: Unresolved Staff Comments",
    "1C": "Item 1C: Cybersecurity",
    "2": "Item 2: Properties",
    "3": "Item 3: Legal Proceedings",
    "4": "Item 4: Mine Safety Disclosures",
    "5": "Item 5: Market",
    "6": "Item 6: Reserved",
    "7": "Item 7: MD&A",
    "7A": "Item 7A: Market Risk",
    "8": "Item 8: Financial Statements",
    "9": "Item 9: Changes",
    "9A": "Item 9A: Controls and Procedures",
    "10": "Item 10: Directors",
    "11": "Item 11: Compensation",
    "12": "Item 12: Ownership",
    "13": "Item 13: Relationships",
    "14": "Item 14: Accountant Fees",
    "15": "Item 15: Exhibits",
}


# ------------------------------------------------------------------
# Page object
# ------------------------------------------------------------------

@dataclass
class ParsedPage:
    text: str
    page_num: int
    doc_id: str
    doc_type: str = "10-K"
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def page(self) -> int:
        return self.page_num


# ------------------------------------------------------------------
# CLEAN HTML
# ------------------------------------------------------------------

_IXBRL_METADATA_TAGS = {"ix:header", "ix:hidden", "ix:references", "ix:resources"}

_BLOCK_ELEMENTS = ["p", "div", "li", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"]


def clean_html(html: str):
    soup = _get_soup(html)

    for tag in soup.find_all(["script", "style", "meta", "link", "noscript", "head"]):
        tag.decompose()

    # iXBRL metadata containers must be removed entirely — their text content is
    # XBRL context data (CIK numbers, member names, period refs), not document prose.
    # Unwrapping them (the generic path below) would dump ~50k chars of metadata noise
    # into the extracted text, collapsing section detection and inflating chunk count.
    for tag_name in _IXBRL_METADATA_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Unwrap remaining iXBRL inline elements (ix:nonFraction, ix:nonNumeric,
    # ix:continuation, etc.) so their text content flows into the surrounding prose.
    for tag in soup.find_all(True):
        name = tag.name or ""
        prefix = tag.prefix or (name.split(":")[0] if ":" in name else None)
        if prefix and prefix.lower() in XBRL_NAMESPACES:
            tag.unwrap()

    return soup


# ------------------------------------------------------------------
# EXTRACT TEXT
# ------------------------------------------------------------------

def _extract_text(soup: BeautifulSoup) -> str:
    parts = []
    body = soup.find("body") or soup

    # Collect only leaf-level block elements to avoid duplicating text that
    # also appears in their ancestor divs/tds.
    for el in body.find_all(["p", "div", "li", "td", "th"]):
        if not el.find(_BLOCK_ELEMENTS):
            text = el.get_text(" ", strip=True)
            if text:
                parts.append(text)

    raw = "\n".join(parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)

    return raw.strip()


# ------------------------------------------------------------------
# SECTION SPLITTING (FIXED)
# ------------------------------------------------------------------

def extract_sections(text: str) -> List[dict]:
    matches = list(ITEM_PATTERN.finditer(text))

    if not matches:
        logger.warning("No SEC items detected — fallback to single section")
        return [{"title": "unknown", "text": text}]

    sections = []
    for i, match in enumerate(matches):
        code = match.group(1).upper()
        title = SEC_SECTIONS.get(code, f"Item {code}")

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        if len(section_text) > 100:
            sections.append({"title": title, "text": section_text})

    # iXBRL documents (e.g. INTC) often embed "Item X." labels only in the
    # table-of-contents, not in the body section headings. When the total text
    # captured by detected sections covers less than 15% of the full document,
    # the match set is almost certainly TOC-only. Fall back to single-section
    # chunking so the entire body content is preserved.
    covered = sum(len(s["text"]) for s in sections)
    if covered < 0.15 * len(text):
        logger.warning(
            "Section detection captured only %.1f%% of document text — "
            "likely TOC-only match. Falling back to single section.",
            100 * covered / max(len(text), 1),
        )
        return [{"title": "unknown", "text": text}]

    logger.info(f"Extracted {len(sections)} sections")
    return sections


# ------------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 2800) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 80:
            chunks.append(chunk.strip())

        start = end

    return chunks


# ------------------------------------------------------------------
# MAIN API
# ------------------------------------------------------------------

def parse_html_filing(
    path: str,
    doc_id: str,
    doc_type: str = "10-K",
) -> List[ParsedPage]:

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    logger.info(f"Parsing HTML filing: {path}")

    soup = clean_html(html)
    text = _extract_text(soup)
    sections = extract_sections(text)

    pages = []
    page_num = 0

    for sec in sections:
        chunks = chunk_text(sec["text"])

        for chunk in chunks:
            pages.append(
                ParsedPage(
                    text=chunk,
                    page_num=page_num,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    section=sec["title"],  
                )
            )
            page_num += 1

    logger.info(f"{len(pages)} chunks created")

    return pages


# ------------------------------------------------------------------
# BASE PARSER ADAPTER
# ------------------------------------------------------------------

class HTMLParser:
    """
    Adapts parse_html_filing to the BaseParser interface expected by Ingest.

    Each ParsedPage produced by the HTML parser is mapped to a models.Page
    so it can flow through the standard Chunker / Validator / SectionDetector
    stages unchanged.
    """

    def extract(self, path: str, doc_id: str) -> list:
        from ingestion.models import Page as ModelPage

        parsed_pages = parse_html_filing(path, doc_id=doc_id)

        if not parsed_pages:
            raise ValueError(f"No content extracted from HTML: {path!r}")

        return [
            ModelPage(
                page=i + 1,
                text=pp.text,
                doc_id=doc_id,
                section=pp.section,
            )
            for i, pp in enumerate(parsed_pages)
        ]