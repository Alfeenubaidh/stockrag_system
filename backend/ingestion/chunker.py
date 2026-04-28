import re
import logging
from typing import Optional, List

from ingestion.doc_metadata import DocumentMetadata
from ingestion.models import Chunk, Page

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP_SENTENCES = 1


# ---------------------------------------------------------------
# CHUNKER
# ---------------------------------------------------------------

class Chunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ):
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")

        self._chunk_size = chunk_size
        self._overlap_sentences = overlap_sentences

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def chunk(
        self,
        pages: List[Page],
        doc_id: str,
        metadata: Optional[DocumentMetadata] = None,
    ) -> List[Chunk]:

        chunks: List[Chunk] = []
        chunk_index = 0

        ticker = metadata.ticker if metadata else None
        doc_type = metadata.doc_type if metadata else None
        filing_date = metadata.filing_date if metadata else None
        accession_number = metadata.accession_number if metadata else None

        # ---------------------------------------------------------------
        # Flatten pages
        # ---------------------------------------------------------------

        full_text = ""
        page_map = []  # (start, end, page_obj)

        for page in pages:
            start = len(full_text)
            full_text += page.text + "\n"
            end = len(full_text)
            page_map.append((start, end, page))

        # ---------------------------------------------------------------
        # Sentence splitting
        # ---------------------------------------------------------------

        sentences = self._split_sentences(full_text)

        cursor = 0
        buffer_sentences: List[str] = []
        buffer_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence) + 1

            if buffer_length + sentence_len > self._chunk_size:
                if buffer_sentences:
                    chunk_text = " ".join(buffer_sentences).strip()

                    if self._should_keep(chunk_text, chunks):
                        page_obj = self._map_to_page(cursor - buffer_length, page_map)

                        # 🔥 FIX: preserve parser section
                        section = page_obj.section or "unknown"

                        chunks.append(
                            Chunk(
                                doc_id=doc_id,
                                chunk_index=chunk_index,
                                text=chunk_text,
                                start_page=page_obj.page,
                                end_page=page_obj.page,
                                ticker=ticker,
                                doc_type=doc_type,
                                filing_date=filing_date,
                                section=section,
                                accession_number=accession_number,
                            )
                        )

                        chunk_index += 1

                    # overlap
                    buffer_sentences = buffer_sentences[-self._overlap_sentences :]
                    buffer_length = sum(len(s) + 1 for s in buffer_sentences)

            buffer_sentences.append(sentence)
            buffer_length += sentence_len
            cursor += sentence_len

        # ---------------------------------------------------------------
        # Final chunk
        # ---------------------------------------------------------------

        if buffer_sentences:
            chunk_text = " ".join(buffer_sentences).strip()

            if self._should_keep(chunk_text, chunks):
                page_obj = self._map_to_page(cursor - buffer_length, page_map)

                # 🔥 FIX: preserve parser section
                section = page_obj.section or "unknown"

                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_page=page_obj.page,
                        end_page=page_obj.page,
                        ticker=ticker,
                        doc_type=doc_type,
                        filing_date=filing_date,
                        section=section,
                        accession_number=accession_number,
                    )
                )

        logger.info(
            "Chunker: %d pages → %d chunks (doc_id=%r)",
            len(pages),
            len(chunks),
            doc_id,
        )

        return chunks

    # ------------------------------------------------------------------
    # FILTERING LOGIC
    # ------------------------------------------------------------------

    def _should_keep(self, text: str, chunks: List[Chunk]) -> bool:
        if self._is_low_value(text):
            return False

        if not self._is_valid_chunk(text):
            return False

        prev_text = chunks[-1].text if chunks else None
        if self._is_duplicate(text, prev_text):
            return False

        return True

    def _is_low_value(self, text: str) -> bool:
        text_upper = text.upper()

        blacklist = [
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
            "FORM 10-K",
            "COMMISSION FILE NUMBER",
            "WASHINGTON, D.C.",
            "INDICATE BY CHECK MARK",
            "SECURITIES REGISTERED PURSUANT",
            "LARGE ACCELERATED FILER",
            "EXCHANGE ACT OF 1934",
            "REGISTRANT",
        ]

        if any(b in text_upper for b in blacklist):
            return True

        if text_upper.count("YES") > 2 and text_upper.count("NO") > 2:
            return True

        return False

    def _is_valid_chunk(self, text: str) -> bool:
        text = text.strip()

        if len(text) < 120:
            return False

        if re.match(r"^[A-Za-z]{1,10}$", text):
            return False

        if text.endswith(("-", "\n")):
            return False

        return True

    def _is_duplicate(self, text: str, prev_text: Optional[str]) -> bool:
        if not prev_text:
            return False

        words_a = set(text.split())
        words_b = set(prev_text.split())

        overlap = len(words_a & words_b)
        ratio = overlap / max(len(words_a), 1)

        return ratio > 0.7

    # ------------------------------------------------------------------
    # SENTENCE SPLITTING
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        protected_abbreviations = [
            "U.S.", "U.K.", "Inc.", "Ltd.", "No.", "Mr.", "Ms.", "Dr.",
            "e.g.", "i.e.", "vs.", "Fig.",
            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
            "Sep.", "Oct.", "Nov.", "Dec.",
        ]

        for abbr in protected_abbreviations:
            text = text.replace(abbr, abbr.replace(".", "<DOT>"))

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace("<DOT>", ".") for s in sentences]

        return sentences

    # ------------------------------------------------------------------
    # PAGE MAPPING
    # ------------------------------------------------------------------

    def _map_to_page(self, idx: int, page_map):
        for start, end, page in page_map:
            if start <= idx < end:
                return page
        return page_map[-1][2]