"""In-memory SearchBackend for tests and offline development.

Naive token-overlap for the keyword arm and cosine over the embedder for the vector
arm — enough to exercise the retriever + RRF fusion without a running OpenSearch.
"""

from __future__ import annotations

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.embeddings.base import Embedder


class InMemoryBackend:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._vectors = {chunk.id: embedder.embed([chunk.text])[0] for chunk in chunks}

    def _passes(self, chunk: Chunk, filters: dict[str, str] | None) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in (filters or {}).items())

    def keyword_ids(self, query: str, *, size: int, filters: dict[str, str] | None) -> list[str]:
        terms = set(query.lower().split())
        scored: list[tuple[int, str]] = []
        for chunk_id, chunk in self._chunks.items():
            if not self._passes(chunk, filters):
                continue
            overlap = len(terms & set(chunk.text.lower().split()))
            if overlap:
                scored.append((overlap, chunk_id))
        scored.sort(reverse=True)
        return [chunk_id for _, chunk_id in scored[:size]]

    def vector_ids(
        self, vector: list[float], *, size: int, filters: dict[str, str] | None
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for chunk_id, candidate in self._vectors.items():
            if not self._passes(self._chunks[chunk_id], filters):
                continue
            scored.append((_dot(vector, candidate), chunk_id))
        scored.sort(reverse=True)
        return [chunk_id for _, chunk_id in scored[:size]]

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in ids if chunk_id in self._chunks]


def _dot(a: list[float], b: list[float]) -> float:
    # HashEmbedder returns unit vectors, so dot product is cosine similarity.
    return sum(x * y for x, y in zip(a, b))
