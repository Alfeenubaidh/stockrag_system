"""
market_data/ratios.py

Fetches key fundamental ratios for a ticker using yfinance.
Fails silently — returns None on any error.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RatioData:
    ticker: str
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    ev_ebitda: Optional[float]
    debt_to_equity: Optional[float]
    revenue_ttm: Optional[int]       # trailing twelve months, USD
    net_income_ttm: Optional[int]
    gross_margin: Optional[float]    # e.g. 0.43 = 43%
    operating_margin: Optional[float]
    return_on_equity: Optional[float]


def get_ratio_data(ticker: str) -> Optional[RatioData]:
    """
    Fetch fundamental ratios for a ticker.
    Returns None if yfinance is unavailable or ticker is invalid.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info

        # If we can't get any useful fundamental data, bail out
        if not info.get("trailingPE") and not info.get("totalRevenue"):
            logger.warning(f"[market_data] No fundamental data found for {ticker}")
            return None

        def _float(key: str) -> Optional[float]:
            val = info.get(key)
            return float(val) if val is not None else None

        def _int(key: str) -> Optional[int]:
            val = info.get(key)
            return int(val) if val is not None else None

        return RatioData(
            ticker=ticker.upper(),
            pe_ratio=_float("trailingPE"),
            forward_pe=_float("forwardPE"),
            ev_ebitda=_float("enterpriseToEbitda"),
            debt_to_equity=_float("debtToEquity"),
            revenue_ttm=_int("totalRevenue"),
            net_income_ttm=_int("netIncomeToCommon"),
            gross_margin=_float("grossMargins"),
            operating_margin=_float("operatingMargins"),
            return_on_equity=_float("returnOnEquity"),
        )

    except ImportError:
        logger.warning("[market_data] yfinance not installed — skipping ratio fetch")
        return None
    except Exception as e:
        logger.warning(f"[market_data] Ratio fetch failed for {ticker}: {e}")
        return None