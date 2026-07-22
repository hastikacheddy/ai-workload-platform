"""In-memory Cache for tests and offline dev. Tracks hits/misses for assertions."""

from __future__ import annotations

import time


class InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if expires_at and expires_at < time.monotonic():
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else 0.0
        self._store[key] = (expires_at, value)
