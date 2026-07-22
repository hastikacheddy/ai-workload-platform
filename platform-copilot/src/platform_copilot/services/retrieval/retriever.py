"""Hybrid retrieval: run both arms, fuse with RRF, hydrate the top-k chunks."""

from __future__ import annotations

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.embeddings.base import Embedder
from platform_copilot.services.retrieval.backend import SearchBackend
from platform_copilot.services.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        backend: SearchBackend,
        embedder: Embedder,
        *,
        candidate_size: int = 20,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._candidate_size = candidate_size

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[Chunk]:
        keyword = self._backend.keyword_ids(query, size=self._candidate_size, filters=filters)
        vector = self._backend.vector_ids(
            self._embedder.embed([query])[0], size=self._candidate_size, filters=filters
        )
        fused = reciprocal_rank_fusion([keyword, vector])
        top_ids = [doc_id for doc_id, _ in fused[:k]]
        by_id = {chunk.id: chunk for chunk in self._backend.get_chunks(top_ids)}
        return [by_id[doc_id] for doc_id in top_ids if doc_id in by_id]
