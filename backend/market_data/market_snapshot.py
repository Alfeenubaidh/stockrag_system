"""
market_data/market_snapshot.py

Assembles price + ratio data into a plain-text context block
suitable for prepending to the LLM generation prompt.

Usage:
    from market_data.market_snapshot import get_snapshot
    block = get_snapshot("AAPL")   # returns str or None
"""

import logging
from typing import Optional

from market_data.prices import get_price_data
from market_data.ratios import get_ratio_data

logger = logging.getLogger(__name__)


def _fmt_currency(val: Optional[int], label: str) -> Optional[str]:
    if val is None:
        return None
    if abs(val) >= 1_000_000_000:
        return f"{label}: ${val / 1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{label}: ${val / 1_000_000:.1f}M"
    return f"{label}: ${val:,}"


def _fmt_ratio(val: Optional[float], label: str, pct: bool = False) -> Optional[str]:
    if val is None:
        return None
    if pct:
        return f"{label}: {val * 100:.1f}%"
    return f"{label}: {val:.2f}x"


def get_snapshot(ticker: str) -> Optional[str]:
    """
    Returns a formatted plain-text market data block for injection into
    the generation context. Returns None if all fetches fail.

    Example output:
        [Market Data — AAPL as of 2026-04-25T14:30:00Z]
        Price: $189.45 USD | Volume: 52,341,200
        52-Week Range: $124.17 – $199.62
        Market Cap: $2.92T
        Valuation: P/E (TTM): 28.4x | Forward P/E: 25.1x | EV/EBITDA: 21.3x
        Leverage: Debt/Equity: 1.47x
        Financials: Revenue (TTM): $383.29B | Net Income (TTM): $96.99B
        Margins: Gross: 45.1% | Operating: 29.8%
        Returns: ROE: 171.9%
    """
    ticker = ticker.upper()
    price_data = get_price_data(ticker)
    ratio_data = get_ratio_data(ticker)

    if price_data is None and ratio_data is None:
        logger.warning(f"[market_data] No data available for {ticker} — skipping snapshot")
        return None

    lines = [f"[Market Data — {ticker} as of {price_data.fetched_at if price_data else 'N/A'}]"]

    # Price block
    if price_data:
        price_line = f"Price: ${price_data.price:,.2f} {price_data.currency}"
        if price_data.volume:
            price_line += f" | Volume: {price_data.volume:,}"
        lines.append(price_line)

        range_line = f"52-Week Range: ${price_data.week_52_low:,.2f} – ${price_data.week_52_high:,.2f}"
        if price_data.market_cap:
            cap = price_data.market_cap
            if cap >= 1_000_000_000_000:
                range_line += f" | Market Cap: ${cap / 1_000_000_000_000:.2f}T"
            elif cap >= 1_000_000_000:
                range_line += f" | Market Cap: ${cap / 1_000_000_000:.2f}B"
            else:
                range_line += f" | Market Cap: ${cap / 1_000_000:.1f}M"
        lines.append(range_line)

    # Valuation
    if ratio_data:
        val_parts = list(filter(None, [
            _fmt_ratio(ratio_data.pe_ratio, "P/E (TTM)"),
            _fmt_ratio(ratio_data.forward_pe, "Forward P/E"),
            _fmt_ratio(ratio_data.ev_ebitda, "EV/EBITDA"),
        ]))
        if val_parts:
            lines.append("Valuation: " + " | ".join(val_parts))

        lev = _fmt_ratio(ratio_data.debt_to_equity, "Debt/Equity")
        if lev:
            lines.append(f"Leverage: {lev}")

        fin_parts = list(filter(None, [
            _fmt_currency(ratio_data.revenue_ttm, "Revenue (TTM)"),
            _fmt_currency(ratio_data.net_income_ttm, "Net Income (TTM)"),
        ]))
        if fin_parts:
            lines.append("Financials: " + " | ".join(fin_parts))

        margin_parts = list(filter(None, [
            _fmt_ratio(ratio_data.gross_margin, "Gross", pct=True),
            _fmt_ratio(ratio_data.operating_margin, "Operating", pct=True),
        ]))
        if margin_parts:
            lines.append("Margins: " + " | ".join(margin_parts))

        roe = _fmt_ratio(ratio_data.return_on_equity, "ROE", pct=True)
        if roe:
            lines.append(f"Returns: {roe}")

    return "\n".join(lines)