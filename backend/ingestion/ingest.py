"""
ingest.py — Orchestrates a single document through the ingestion pipeline.

Pipeline:
    PDFParser → TextCleaner → Chunker → ChunkValidator → SectionDetector
"""

import logging
from typing import Optional, List

from ingestion.chunker import Chunker
from ingestion.chunk_validator import ChunkValidator
from ingestion.doc_metadata import DocumentMetadata
from ingestion.models import Chunk
from ingestion.pdf_parser import PDFParser
from ingestion.section_detector import SectionDetector
from ingestion.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class Ingest:
    def __init__(
        self,
        doc_id: str,
        metadata: Optional[DocumentMetadata] = None,
        parser: Optional[PDFParser] = None,
        cleaner: Optional[TextCleaner] = None,
        chunker: Optional[Chunker] = None,
        validator: Optional[ChunkValidator] = None,
        section_detector: Optional[SectionDetector] = None,
    ):
        self._doc_id = doc_id
        self._metadata = metadata

        self._parser = parser or PDFParser()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or Chunker()
        self._validator = validator or ChunkValidator()
        self._section_detector = section_detector or SectionDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, path: str) -> List[Chunk]:
        logger.info("Ingest.run: starting doc_id=%r path=%r", self._doc_id, path)

        # 1. Extract
        pages = self._parser.extract(path, doc_id=self._doc_id)

        if not pages:
            raise ValueError("No pages parsed")

        # 2. Clean
        clean_pages = self._cleaner.clean(pages)

        # 3. Chunk
        raw_chunks = self._chunker.chunk(
            clean_pages,
            doc_id=self._doc_id,
            metadata=self._metadata,
        )

        # -------------------------------
        # Integrity checks
        # -------------------------------
        if any(c.doc_id != self._doc_id for c in raw_chunks):
            raise ValueError("Chunk doc_id mismatch")

        if any(not c.text.strip() for c in raw_chunks):
            raise ValueError("Empty chunks detected")

        expected = list(range(len(raw_chunks)))
        actual = [c.chunk_index for c in raw_chunks]
        if actual != expected:
            raise ValueError("Chunk indices not sequential")

        # -------------------------------
        # 4. Validate (WITH OBSERVABILITY)
        # -------------------------------
        before = len(raw_chunks)

        filtered_chunks = self._validator.validate(raw_chunks)

        after = len(filtered_chunks)

        logger.info(
            "ChunkValidator — kept %d/%d (%.2f%%)",
            after,
            before,
            (after / before * 100) if before else 0,
        )

        # -------------------------------
        # 5. Section detection
        # -------------------------------
        chunks = self._section_detector.assign(filtered_chunks)

        assigned = sum(1 for c in chunks if c.section is not None)

        logger.info(
            "Ingest.run: finished doc_id=%r — %d chunks (%d with sections)",
            self._doc_id,
            len(chunks),
            assigned,
        )

        return chunks