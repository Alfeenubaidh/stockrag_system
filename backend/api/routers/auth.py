from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import settings

logger = logging.getLogger(__name__)

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache(maxsize=1)
def _valid_keys() -> frozenset[str]:
    """Parse API_KEYS once and cache. Empty → auth disabled."""
    raw = settings.api_keys.strip()
    if not raw:
        return frozenset()
    keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
    logger.info("Auth enabled: %d API key(s) loaded", len(keys))
    return keys


def require_api_key(key: str | None = Security(_header_scheme)) -> None:
    """
    FastAPI dependency. Inject into any route that must be protected.

    Behaviour:
    - API_KEYS unset / empty → auth disabled, all requests pass through.
    - API_KEYS set           → X-API-Key header required; 401 on missing or invalid key.
    """
    valid = _valid_keys()
    if not valid:
        return  # auth disabled in dev

    if not key or key not in valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
