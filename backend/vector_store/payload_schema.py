from __future__ import annotations

from typing import TypedDict, Dict, Any
import datetime
import re


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ChunkPayload(TypedDict):
    # --- provenance ---
    doc_id: str
    chunk_index: int
    ticker: str
    doc_type: str
    filing_date: str
    filing_timestamp: int
    section: str
    accession_number: str

    # --- position ---
    start_page: int
    end_page: int

    # --- content ---
    text: str


# ---------------------------------------------------------------------------
# Required + Indexed Fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: tuple[str, ...] = (
    "doc_id",
    "chunk_index",
    "ticker",
    "doc_type",
    "filing_date",
    "section",
    "accession_number",
    "start_page",
    "end_page",
    "text",
)

INDEXED_FIELDS: tuple[str, ...] = (
    "ticker",
    "doc_type",
    "section",
    "filing_timestamp",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_timestamp(date_str: str) -> int:
    if not date_str or date_str == "unknown":
        return -1
    try:
        dt = datetime.datetime.fromisoformat(date_str)
        return int(dt.timestamp())
    except Exception:
        return -1


def normalize_ticker(t: str | None) -> str:
    if not t:
        return ""
    t = t.strip().upper()
    if ":" in t:
        t = t.split(":")[-1]
    if "." in t:
        t = t.split(".")[0]
    return t


def normalize_section(section: str | None) -> str:
    if not section:
        return "unknown"

    s = section.strip().lower()

    # normalize SEC prefixes
    s = re.sub(r"^item\s+\d+[a-z]?\s*[:.]?\s*", "", s)

    if not s:
        return "unknown"

    return s


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Validation + Conversion
# ---------------------------------------------------------------------------

def chunk_to_payload(chunk: Dict[str, Any]) -> ChunkPayload:
    missing = [f for f in REQUIRED_FIELDS if f not in chunk]
    if missing:
        raise KeyError(
            f"Missing required fields: {missing} | doc_id={chunk.get('doc_id')}"
        )

    try:
        filing_date = str(chunk.get("filing_date", "unknown")).strip()

        ticker = normalize_ticker(chunk.get("ticker"))
        section = normalize_section(chunk.get("section"))
        text = clean_text(chunk.get("text"))

        payload: ChunkPayload = {
            # --- provenance ---
            "doc_id": str(chunk["doc_id"]).strip(),
            "chunk_index": int(chunk["chunk_index"]),
            "ticker": ticker,
            "doc_type": str(chunk["doc_type"]).strip(),
            "filing_date": filing_date,
            "filing_timestamp": _to_timestamp(filing_date),
            "section": section,
            "accession_number": str(chunk["accession_number"]).strip(),

            # --- position ---
            "start_page": int(chunk["start_page"]),
            "end_page": int(chunk["end_page"]),

            # --- content ---
            "text": text,
        }

    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Type casting failed | doc_id={chunk.get('doc_id')} | {e}"
        ) from e

    # -----------------------------------------------------------------------
    # Sanity checks (STRICT only where it matters)
    # -----------------------------------------------------------------------

    if not payload["doc_id"]:
        raise ValueError("doc_id cannot be empty")

    if payload["chunk_index"] < 0:
        raise ValueError("chunk_index must be >= 0")

    if not payload["ticker"]:
        raise ValueError("ticker cannot be empty")

    if not payload["text"]:
        raise ValueError("text cannot be empty")

    if payload["start_page"] > payload["end_page"]:
        raise ValueError("start_page > end_page")

    if payload["filing_date"] != "unknown" and len(payload["filing_date"]) < 8:
        raise ValueError("invalid filing_date")

    # 🔥 Relaxed constraint (important fix)
    if len(payload["text"]) < 40:
        raise ValueError("chunk too small")

    if len(payload["text"]) > 8000:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"Oversized chunk ({len(payload['text'])} chars) | doc_id={payload['doc_id']} "
            f"chunk={payload['chunk_index']}"
        )

    return payload