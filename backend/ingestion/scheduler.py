"""
ingestion/scheduler.py — Nightly EDGAR filing check and auto-ingest.

Delegates all EDGAR fetch/download/ingest logic to data_sources.sec_edgar.
Checks each ticker in settings.watchlist_tickers for new 10-K and 10-Q
filings published since the previous run.

Schedule: daily at 02:00 local time.
Failure mode: per-ticker try/except — one failure never stops the others.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).parent.parent / "data" / "scheduler_state.json"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-ticker check  (delegates to data_sources.sec_edgar)
# ---------------------------------------------------------------------------

def _check_ticker(ticker: str, since: str) -> None:
    logger.info("Checking %s...", ticker)

    from data_sources.sec_edgar import fetch_and_ingest, get_latest_filing

    for doc_type in ("10-K", "10-Q"):
        filing = get_latest_filing(ticker, doc_type)
        if filing is None:
            logger.info("No new filings for %s (%s)", ticker, doc_type)
            continue

        if filing["filing_date"] < since:
            logger.info("No new filings for %s", ticker)
            continue

        label = f"{ticker} {doc_type} {filing['filing_date']}"
        logger.info("New filing found: %s", label)

        ok = fetch_and_ingest(ticker, doc_type)
        if ok:
            logger.info("Ingested %s", label)
        else:
            logger.info("Skipped %s (already ingested or no PDF)", label)


# ---------------------------------------------------------------------------
# Nightly job
# ---------------------------------------------------------------------------

def _nightly_job() -> None:
    from config.settings import settings

    tickers: list[str] = getattr(settings, "watchlist_tickers", [])
    if not tickers:
        logger.info("watchlist_tickers is empty — nothing to check")
        return

    since = (date.today() - timedelta(days=2)).isoformat()

    state = _load_state()
    state["last_run"] = date.today().isoformat()

    for ticker in tickers:
        try:
            _check_ticker(ticker.upper(), since)
        except Exception as exc:
            logger.error("Scheduler error for %s: %s", ticker, exc, exc_info=True)

    _save_state(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()
    scheduler.add_job(_nightly_job, "cron", hour=2, minute=0, id="nightly_edgar_check")
    scheduler.start()
    logger.info("Scheduler started — nightly EDGAR check at 02:00")
    return scheduler
