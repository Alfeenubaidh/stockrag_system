import logging

logger = logging.getLogger(__name__)


class DataValidator:

    def validate(self, pages, chunks):
        self._check_empty_sections(pages)
        self._check_chunk_sizes(chunks)
        self._check_null_sections(chunks)

    def _check_empty_sections(self, pages):
        if len(pages) > 10:
            if all(getattr(p, "section", None) is None for p in pages[10:]):
                raise ValueError("No sections detected after page 10")

    def _check_chunk_sizes(self, chunks):
        sizes = [len(c.text) for c in chunks]

        if not sizes:
            raise ValueError("No chunks produced")

        avg = sum(sizes) / len(sizes)

        if avg < 200:
            raise ValueError("Chunks too small (bad split)")

        if avg > 2000:
            raise ValueError("Chunks too large (bad split)")

    def _check_null_sections(self, chunks):
        nulls = sum(1 for c in chunks if c.section is None)

        if nulls / max(len(chunks), 1) > 0.6:
            logger.warning("High null section ratio: %.2f", nulls / len(chunks))