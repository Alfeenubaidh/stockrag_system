from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None
_unavailable = False  # latched after first failure — avoids log spam


def _get_redis():
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is not None:
        return _client
    try:
        import redis
        from config.settings import settings

        redis_url = getattr(settings, "redis_url", "redis://localhost:6379")
        _client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        _client.ping()
        return _client
    except Exception as exc:
        _unavailable = True
        logger.warning("Redis unavailable — skipping cache: %s", exc)
        return None


def cache_key(query: str, ticker: Optional[str], date_range: Optional[str]) -> str:
    """SHA-256 of (query + ticker + date_range) → hex digest."""
    raw = f"{query}|{ticker or ''}|{date_range or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str) -> Optional[str]:
    r = _get_redis()
    if r is None:
        return None
    try:
        value = r.get(key)
        if value is not None:
            logger.debug("Cache hit for key %s", key[:16])
            return value
        logger.debug("Cache miss")
        return None
    except Exception as exc:
        logger.warning("Cache get error: %s", exc)
        return None


def set_cached(key: str, value: str, ttl_seconds: int = 3600) -> None:
    r = _get_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl_seconds)
    except Exception as exc:
        logger.warning("Cache set error: %s", exc)
