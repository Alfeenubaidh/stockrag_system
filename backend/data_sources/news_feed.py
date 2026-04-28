"""
data_sources/news_feed.py — Fetch recent news for a ticker via Yahoo Finance RSS.

Items are stored in Qdrant with doc_type="news", section="news" so they can
be retrieved alongside SEC filings when queries mention recent events.
Live news is NEVER embedded as historical fact — items carry their
published date and the generation layer must treat them as current context.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_news(ticker: str, max_items: int = 20) -> list[dict]:
    """
    Fetch up to *max_items* recent news items for *ticker* from Yahoo Finance RSS.

    Returns a list of::

        {
            "ticker":    "AAPL",
            "title":     "Apple Reports Record Quarter",
            "summary":   "...",
            "published": "2024-11-01",
            "url":       "https://...",
        }

    Returns [] on any error (fail-silent — news is supplementary context,
    never a hard dependency).
    """
    try:
        import feedparser
    except ImportError:
        logger.error("news_feed: feedparser not installed — run: pip install feedparser")
        return []

    url = _RSS_URL.format(ticker=ticker.upper())
    logger.info("news_feed: fetching RSS for %s from %s", ticker, url)

    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        logger.warning("news_feed: feedparser error for %s: %s", ticker, exc)
        return []

    if feed.get("bozo") and not feed.get("entries"):
        logger.warning("news_feed: malformed feed for %s", ticker)
        return []

    items: list[dict] = []
    for entry in feed.get("entries", [])[:max_items]:
        published = _parse_date(entry)
        items.append(
            {
                "ticker":    ticker.upper(),
                "title":     entry.get("title", "").strip(),
                "summary":   _clean_summary(entry.get("summary", "")),
                "published": published,
                "url":       entry.get("link", ""),
            }
        )

    logger.info("news_feed: fetched %d items for %s", len(items), ticker)
    return items


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_news(ticker: str, max_items: int = 20) -> int:
    """
    Fetch news for *ticker* and upsert each item as a chunk in Qdrant.

    Uses ``doc_type="news"`` and ``section="news"`` so retrieval filters
    can include or exclude news independently of SEC filings.

    Returns the count of items successfully upserted (0 on any failure).
    """
    items = fetch_news(ticker, max_items=max_items)
    if not items:
        return 0

    try:
        from config.settings import settings
        from embeddings.embedder import EmbeddingConfig, EmbeddingPipeline
        from vector_store.qdrant_client import get_qdrant_client
        from vector_store.qdrant_store import QdrantStore

        qdrant = get_qdrant_client()
        store = QdrantStore(client=qdrant, collection_name=settings.qdrant_collection)
        embedder = EmbeddingPipeline(config=EmbeddingConfig(batch_size=128))
    except Exception as exc:
        logger.error("news_feed: failed to initialise Qdrant/embedder: %s", exc)
        return 0

    chunk_dicts: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        text = _build_chunk_text(item)
        doc_id = f"news_{ticker.upper()}_{item['published']}"
        chunk_dicts.append(
            {
                "chunk_id":         f"{doc_id}_{idx}",
                "doc_id":           doc_id,
                "ticker":           ticker.upper(),
                "doc_type":         "news",
                "filing_date":      item["published"],
                "accession_number": str(uuid.uuid5(uuid.NAMESPACE_URL, item["url"])),
                "section":          "news",
                "start_page":       0,
                "end_page":         0,
                "chunk_index":      idx,
                "text":             text,
            }
        )

    if not chunk_dicts:
        return 0

    try:
        embedded = embedder.embed_chunks(chunk_dicts)
        store.upsert_chunks(embedded)
        logger.info("news_feed: upserted %d news chunks for %s", len(embedded), ticker)
        return len(embedded)
    except Exception as exc:
        logger.error("news_feed: upsert failed for %s: %s", ticker, exc)
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(entry: Any) -> str:
    """Return ISO date string from a feedparser entry, defaulting to today."""
    for attr in ("published_parsed", "updated_parsed"):
        t = entry.get(attr)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clean_summary(raw: str) -> str:
    """Strip HTML tags from RSS summary text."""
    import re
    return re.sub(r"<[^>]+>", "", raw).strip()


def _build_chunk_text(item: dict) -> str:
    parts = [item["title"]]
    if item["summary"] and item["summary"] != item["title"]:
        parts.append(item["summary"])
    if item["url"]:
        parts.append(f"Source: {item['url']}")
    return "\n".join(parts)
