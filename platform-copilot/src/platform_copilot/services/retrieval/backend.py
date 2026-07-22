"""The retrieval boundary.

The retriever depends on this Protocol, not on OpenSearch, so the live cluster and
the in-memory test backend are interchangeable. Each arm returns chunk ids; the
retriever fuses and hydrates them.
"""

from __future__ import annotations

from typing import Protocol

from platform_copilot.schemas.chunk import Chunk


class SearchBackend(Protocol):
    def keyword_ids(
        self, query: str, *, size: int, filters: dict[str, str] | None
    ) -> list[str]: ...

    def vector_ids(
        self, vector: list[float], *, size: int, filters: dict[str, str] | None
    ) -> list[str]: ...

    def get_chunks(self, ids: list[str]) -> list[Chunk]: ...
