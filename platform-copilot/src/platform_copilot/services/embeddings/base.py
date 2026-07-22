"""The embedding boundary.

Retrieval depends on this Protocol, not on a concrete provider, so a Jina API
client, a local sentence-transformer, or the test fake are all interchangeable.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
