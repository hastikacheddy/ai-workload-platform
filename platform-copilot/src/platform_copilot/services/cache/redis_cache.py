"""Redis-backed Cache.

INTEGRATION-ONLY: needs a running Redis, so it is used when the stack is up (M5) and
is not exercised by the offline tests. Implements the same Cache Protocol as the fake.
"""

from __future__ import annotations

import redis


class RedisCache:
    def __init__(self, url: str) -> None:
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        value: str | None = self._client.get(key)
        return value

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        self._client.set(key, value, ex=ttl_seconds or None)
