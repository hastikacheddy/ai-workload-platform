"""Live OpenSearch retrieval backend.

INTEGRATION-ONLY: needs a running OpenSearch cluster, so it is exercised once the
Docker stack is up (M2/M3), not in the offline unit tests. It implements the same
SearchBackend Protocol the in-memory fake does, so nothing downstream changes.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.opensearch.index import index_settings
from platform_copilot.services.opensearch.query import bm25_query, knn_query


class OpenSearchBackend:
    def __init__(self, client: OpenSearch, index: str) -> None:
        self._client = client
        self._index = index

    def ensure_index(self, embedding_dim: int) -> None:
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(index=self._index, body=index_settings(embedding_dim))

    def index_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        actions: list[dict[str, Any]] = [
            {
                "_index": self._index,
                "_id": chunk.id,
                "_source": {
                    "doc_slug": chunk.doc_slug,
                    "ordinal": chunk.ordinal,
                    "heading_path": chunk.heading_path,
                    "text": chunk.text,
                    "source_type": chunk.metadata.get("source_type"),
                    "service": chunk.metadata.get("service"),
                    "severity": chunk.metadata.get("severity"),
                    "metadata": chunk.metadata,
                    "embedding": vector,
                },
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        bulk(self._client, actions)

    def keyword_ids(self, query: str, *, size: int, filters: dict[str, str] | None) -> list[str]:
        body = bm25_query(query, filters=filters, size=size)
        hits = self._client.search(index=self._index, body=body)["hits"]["hits"]
        return [hit["_id"] for hit in hits]

    def vector_ids(
        self, vector: list[float], *, size: int, filters: dict[str, str] | None
    ) -> list[str]:
        body = knn_query(vector, filters=filters, size=size)
        hits = self._client.search(index=self._index, body=body)["hits"]["hits"]
        return [hit["_id"] for hit in hits]

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        docs = self._client.mget(index=self._index, body={"ids": ids})["docs"]
        chunks: list[Chunk] = []
        for doc in docs:
            if doc.get("found"):
                source = doc["_source"]
                chunks.append(
                    Chunk(
                        doc_slug=source["doc_slug"],
                        ordinal=source["ordinal"],
                        heading_path=source["heading_path"],
                        text=source["text"],
                        metadata=source.get("metadata", {}),
                    )
                )
        return chunks
