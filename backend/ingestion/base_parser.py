from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ingestion.models import Page


class BaseParser(ABC):
    """
    Abstract parser contract for document ingestion.

    Responsibilities:
    - Extract raw text
    - Preserve page boundaries
    - Attach provided doc_id to each Page

    Must NOT:
    - Clean text
    - Chunk text
    - Filter content
    """

    @abstractmethod
    def extract(self, path: Path, doc_id: str) -> List[Page]:
        """
        Extract raw text into Page objects.
        """
        ...

    # ------------------------------------------------------------------
    # Optional validation helper
    # ------------------------------------------------------------------

    def _validate_output(self, pages: List[Page], doc_id: str) -> None:
        if not pages:
            raise ValueError("Parser returned no pages")

        for p in pages:
            if p.doc_id != doc_id:
                raise ValueError("Page doc_id mismatch")

            if p.page <= 0:
                raise ValueError("Invalid page number")

            if not p.text.strip():
                raise ValueError("Empty page text")