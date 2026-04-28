import re
import logging
from dataclasses import dataclass, field
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

def clean_html(html: str):
    soup = _get_soup(html)

    for tag in soup.find_all(["script", "style", "meta", "link", "noscript", "head"]):
        tag.decompose()

    for tag in soup.find_all(True):
        name = tag.name or ""
        prefix = tag.prefix or (name.split(":")[0] if ":" in name else None)
        if prefix and prefix.lower() in XBRL_NAMESPACES:
            tag.unwrap()

    return soup


# ------------------------------------------------------------------
# EXTRACT TEXT (FIXED: preserve structure)
# ------------------------------------------------------------------

def _extract_text(soup: BeautifulSoup) -> str:
    parts = []

    for el in soup.find_all(["p", "div", "span", "li", "td", "th"]):
        text = el.get_text(" ", strip=True)
        if text:
            parts.append(text)

    # 🔥 CRITICAL: preserve line boundaries
    raw = "\n".join(parts)

    # normalize spacing WITHOUT destroying structure
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)

    return raw.strip()


# ------------------------------------------------------------------
# SECTION SPLITTING (FIXED)
# ------------------------------------------------------------------

def extract_sections(text: str) -> List[dict]:
    matches = list(ITEM_PATTERN.finditer(text))

    sections = []

    if not matches:
        logger.warning("No SEC items detected — fallback to single section")
        return [{"title": "unknown", "text": text}]

    for i, match in enumerate(matches):
        code = match.group(1).upper()
        title = SEC_SECTIONS.get(code, f"Item {code}")

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        if len(section_text) > 100:
            sections.append({
                "title": title,
                "text": section_text
            })

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
                    section=sec["title"],  # 🔥 now always valid
                )
            )
            page_num += 1

    logger.info(f"{len(pages)} chunks created")

    return pages