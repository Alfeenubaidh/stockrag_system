"""
market_data/prices.py

Fetches live price data for a ticker using yfinance.
Fails silently — returns None on any error.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    ticker: str
    price: float
    currency: str
    volume: int
    week_52_high: float
    week_52_low: float
    market_cap: Optional[int]
    fetched_at: str  # ISO datetime string


def get_price_data(ticker: str) -> Optional[PriceData]:
    """
    Fetch live price snapshot for a ticker.
    Returns None if yfinance is unavailable or ticker is invalid.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            logger.warning(f"[market_data] No price found for {ticker}")
            return None

        return PriceData(
            ticker=ticker.upper(),
            price=float(price),
            currency=info.get("currency", "USD"),
            volume=int(info.get("regularMarketVolume") or 0),
            week_52_high=float(info.get("fiftyTwoWeekHigh") or 0.0),
            week_52_low=float(info.get("fiftyTwoWeekLow") or 0.0),
            market_cap=info.get("marketCap"),
            fetched_at=datetime.utcnow().isoformat() + "Z",
        )

    except ImportError:
        logger.warning("[market_data] yfinance not installed — skipping price fetch")
        return None
    except Exception as e:
        logger.warning(f"[market_data] Price fetch failed for {ticker}: {e}")
        return None