"""
data_sources/earnings_transcripts.py — Download earnings call transcripts.

Source: Motley Fool (free, public HTML).
Transcripts are saved as plain-text .txt files — the ingest pipeline
handles PDF only, so these are stored for future reference or manual review.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; StockRAG/1.0; research@stockrag.com)"
)
_SEARCH_URL = "https://www.fool.com/earnings-call-transcripts/"
_SLEEP = 1.5  # polite crawl rate for a public web source
_TIMEOUT = 20


def _get(url: str, **kwargs) -> requests.Response:
    time.sleep(_SLEEP)
    return requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=_TIMEOUT,
        **kwargs,
    )


def get_transcript_url(ticker: str, year: int, quarter: int) -> Optional[str]:
    """
    Search Motley Fool's transcript index for *ticker* Q*quarter* *year*.

    Returns the URL of the transcript article page if found, None otherwise.
    Motley Fool does not expose a structured API, so this uses a Google-style
    site: search via their on-site search endpoint.
    """
    query = f"{ticker.upper()} Q{quarter} {year} earnings call transcript"
    search_url = f"https://www.fool.com/search/solr.aspx?q={requests.utils.quote(query)}"

    logger.info("earnings_transcripts: searching for %s", query)
    try:
        resp = _get(search_url)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("earnings_transcripts: search request failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Motley Fool search results contain article links; pick the first result
    # whose URL path contains "earnings-call-transcript"
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "earnings-call-transcript" not in href.lower():
            continue
        # Expect ticker mention + year in the URL slug
        slug = href.lower()
        if ticker.lower() in slug and str(year) in slug:
            full_url = href if href.startswith("http") else f"https://www.fool.com{href}"
            logger.info("earnings_transcripts: found URL %s", full_url)
            return full_url

    logger.info(
        "earnings_transcripts: no transcript found for %s Q%d %d", ticker, quarter, year
    )
    return None


def download_transcript(
    ticker: str,
    year: int,
    quarter: int,
    output_dir: str = "data/raw/pdfs",
) -> Optional[Path]:
    """
    Fetch the transcript HTML page, extract the main article text, and save
    it as ``{TICKER}_{YEAR}_Q{QUARTER}_transcript.txt`` in *output_dir*.

    Returns the Path to the saved file, or None on failure.
    """
    url = get_transcript_url(ticker, year, quarter)
    if url is None:
        return None

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{ticker.upper()}_{year}_Q{quarter}_transcript.txt"
    dest = dest_dir / filename

    if dest.exists():
        logger.info("earnings_transcripts: already exists, skipping — %s", dest)
        return dest

    logger.info("earnings_transcripts: fetching %s", url)
    try:
        resp = _get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("earnings_transcripts: fetch failed for %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Motley Fool article body lives inside <div class="article-body"> or
    # <div id="article-body">; fall back to the largest <article> tag.
    body = (
        soup.find("div", class_="article-body")
        or soup.find("div", id="article-body")
        or soup.find("article")
    )
    if body is None:
        logger.warning("earnings_transcripts: could not locate article body at %s", url)
        return None

    # Strip scripts, styles, and navigation cruft
    for tag in body.find_all(["script", "style", "nav", "aside", "figure"]):
        tag.decompose()

    raw_text = body.get_text(separator="\n")
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    if len(text) < 200:
        logger.warning(
            "earnings_transcripts: extracted text too short (%d chars) — likely blocked",
            len(text),
        )
        return None

    dest.write_text(text, encoding="utf-8")
    logger.info(
        "earnings_transcripts: saved %s (%d chars)", dest, len(text)
    )
    return dest
