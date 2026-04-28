"""
batch_ingest.py — FINAL production-grade ingestion pipeline (FULLY FIXED)

Includes:
- HTML + PDF parsing
- Chunking + validation
- Embedding via embed_chunks()
- Qdrant upsert
"""

import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from ingestion.doc_metadata import MetadataExtractor
from ingestion.pdf_parser import PDFParser
from ingestion.html_parser import parse_html_filing
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker
from ingestion.section_detector import SectionDetector

from ingestion.chunk_validator import ChunkValidator
from ingestion.validator import DataValidator
from ingestion.versioning import VersionManager
from ingestion.hasher import compute_file_hash

# ✅ VECTOR + EMBEDDING
from vector_store.qdrant_client import get_qdrant_client, COLLECTION_NAME
from vector_store.qdrant_store import QdrantStore
from embeddings.embedder import EmbeddingPipeline, EmbeddingConfig

logger = logging.getLogger(__name__)


class BatchIngester:
    def __init__(
        self,
        data_dir: Path | str = Path("data/raw"),
        max_workers: int = 4,
        max_retries: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.max_retries = max_retries

        self.meta_extractor = MetadataExtractor()
        self.validator = ChunkValidator()
        self.data_validator = DataValidator()

        self.version_manager = VersionManager()
        self.version_manager.check_or_initialize()

        # ✅ QDRANT
        self.qdrant_client = get_qdrant_client()
        self.vector_store = QdrantStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
        )

        # ✅ EMBEDDER
        self.embedder = EmbeddingPipeline(
            config=EmbeddingConfig(batch_size=64)
        )

    # ------------------------------------------------------------------
    # STREAM
    # ------------------------------------------------------------------

    def stream(self, overwrite: bool = False):
        files = (
            list(self.data_dir.rglob("*.pdf"))
            + list(self.data_dir.rglob("*.html"))
            + list(self.data_dir.rglob("*.htm"))
        )

        if not files:
            logger.warning("No files found in %s", self.data_dir)
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = deque()

            for file in files:
                futures.append(
                    executor.submit(self._process_with_retry, file)
                )

            while futures:
                future = futures.popleft()

                try:
                    doc_id, chunks = future.result()

                    if chunks:
                        yield doc_id, chunks

                except Exception:
                    logger.exception("Unhandled worker failure")

    # ------------------------------------------------------------------
    # RETRY
    # ------------------------------------------------------------------

    def _process_with_retry(self, file):
        doc_id = file.stem
        attempt = 0

        while attempt <= self.max_retries:
            try:
                chunks = self._process_file(file, doc_id)
                return doc_id, chunks

            except Exception as e:
                attempt += 1
                logger.warning("Retry %d for %s: %s", attempt, doc_id, str(e))

                if attempt > self.max_retries:
                    logger.error("Failed: %s", doc_id)
                    return doc_id, None

                time.sleep(0.5 * (2 ** attempt))

    # ------------------------------------------------------------------
    # WORKER
    # ------------------------------------------------------------------

    def _process_file(self, file: Path, doc_id: str):
        cleaner = TextCleaner()
        chunker = Chunker()
        section_detector = SectionDetector()

        suffix = file.suffix.lower()

        # ---------------- HTML ----------------
        if suffix in (".html", ".htm"):
            pages = parse_html_filing(
                path=str(file),
                doc_id=doc_id,
                doc_type="10-K",
            )

            if not pages:
                raise ValueError("No pages parsed from HTML")

            pages = cleaner.clean(pages)

            metadata = self.meta_extractor.extract(
                text=" ".join(p.text for p in pages[:5]),
                filename=file.name,
            )

            for page in pages:
                page.doc_type = metadata.doc_type

        # ---------------- PDF ----------------
        else:
            parser = PDFParser()
            pages = parser.extract(str(file), doc_id)

            if not pages:
                raise ValueError("No pages parsed")

            pages = cleaner.clean(pages)

            metadata = self.meta_extractor.extract(
                text=" ".join(p.text for p in pages[:10]),
                filename=file.name,
            )

            for page in pages:
                page.doc_type = metadata.doc_type

        # ---------------- CHUNK ----------------
        chunks = chunker.chunk(pages, doc_id, metadata)
        chunks = self.validator.validate(chunks)
        self.data_validator.validate(pages, chunks)

        # ---------------- EMBEDDING ----------------
        chunk_dicts = []

        for c in chunks:
            chunk_dicts.append({
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "ticker": c.ticker,
                "doc_type": c.doc_type,
                "filing_date": c.filing_date,
                "section": c.section,
                "accession_number": c.accession_number,
                "start_page": c.start_page,
                "end_page": c.end_page,
                "text": c.text,
            })

        embedded_chunks = self.embedder.embed_chunks(chunk_dicts)

        # ---------------- QDRANT UPSERT ----------------
        self.vector_store.upsert_chunks(embedded_chunks)

        logger.info(
            "Upserted %s → %d chunks",
            doc_id,
            len(embedded_chunks),
        )

        return chunks
