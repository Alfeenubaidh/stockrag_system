from dataclasses import dataclass
from typing import Optional


@dataclass
class Page:
    page: int
    text: str
    doc_id: str
    section: Optional[str] = None


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int  # must be per-document

    text: str

    # FIXED: page range instead of single page
    start_page: int
    end_page: int

    # Metadata
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    filing_date: Optional[str] = None
    section: Optional[str] = None
    accession_number: Optional[str] = None

    # OPTIONAL but high-value
    start_char: Optional[int] = None
    end_char: Optional[int] = None