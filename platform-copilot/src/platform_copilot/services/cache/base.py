"""The cache boundary — Redis in production, an in-memory dict in tests."""

from __future__ import annotations

from typing import Protocol


class Cache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None: ...
