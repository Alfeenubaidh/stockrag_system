from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

RATE_LIMIT = "60/minute"


def _make_limiter() -> Limiter:
    """
    Try Redis-backed storage first (shared state across workers).
    Fall back to in-memory silently — requests are never blocked due to
    storage unavailability (fail-open).
    """
    try:
        import redis
        from config.settings import settings

        redis_url = getattr(settings, "redis_url", "redis://localhost:6379")
        redis.Redis.from_url(redis_url, socket_connect_timeout=1).ping()
        logger.info("Rate limiter: Redis storage at %s", redis_url)
        return Limiter(key_func=get_remote_address, storage_uri=redis_url)
    except Exception:
        logger.info("Rate limiter: Redis unavailable — using in-memory storage (fail open)")
        return Limiter(key_func=get_remote_address)


limiter = _make_limiter()
