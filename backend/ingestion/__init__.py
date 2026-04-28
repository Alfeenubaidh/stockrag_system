"""
ingestion package — PDF ingestion for StockRAG.

Public surface:
    Ingest            — single-document pipeline entry point
    BatchIngester     — batch pipeline entry point (streams results)
    PDFParser         — swap in for testing or OCR variants
    TextCleaner       — extend with custom cleaning passes
    Chunker           — configure chunk/overlap sizes
    ChunkValidator    — extend with custom filters
    SectionDetector   — post-validation section assignment
    BaseParser        — subclass for new parser backends
    DocumentMetadata  — source metadata dataclass
    Page, Chunk       — shared data models
"""

from ingestion.models import Page, Chunk
from ingestion.ingest import Ingest
from ingestion.batch_ingest import BatchIngester
from ingestion.pdf_parser import PDFParser
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker
from ingestion.chunk_validator import ChunkValidator
from ingestion.section_detector import SectionDetector
from ingestion.base_parser import BaseParser
from ingestion.doc_metadata import DocumentMetadata, MetadataExtractor

__all__ = [
    "Page",
    "Chunk",
    "Ingest",
    "BatchIngester",
    "PDFParser",
    "TextCleaner",
    "Chunker",
    "ChunkValidator",
    "SectionDetector",
    "BaseParser",
    "DocumentMetadata",
    "MetadataExtractor",
]