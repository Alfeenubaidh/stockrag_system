from __future__ import annotations

from datetime import datetime
from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range


def build_filters(
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    section: Optional[str] = None,
) -> Optional[Filter]:
    """
    Build a Qdrant Filter combining must conditions for every non-None arg.

    fiscal_year maps to filing_timestamp range (Jan 1 00:00 – Dec 31 23:59:59
    of that calendar year, stored as UTC unix seconds).
    """
    must = []

    if ticker:
        must.append(FieldCondition(key="ticker", match=MatchValue(value=ticker.upper())))

    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

    if fiscal_year is not None:
        start_ts = int(datetime(fiscal_year, 1, 1, 0, 0, 0).timestamp())
        end_ts = int(datetime(fiscal_year, 12, 31, 23, 59, 59).timestamp())
        must.append(
            FieldCondition(
                key="filing_timestamp",
                range=Range(gte=start_ts, lte=end_ts),
            )
        )

    if section:
        must.append(FieldCondition(key="section", match=MatchValue(value=section)))

    return Filter(must=must) if must else None
