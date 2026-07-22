"""Request bodies for the two retrieval arms.

Each arm returns its own ranked list of chunk ids; the app fuses them with
Reciprocal Rank Fusion (``services.retrieval.fusion``) rather than relying on
OpenSearch to combine incomparable score scales.
"""

from __future__ import annotations

from typing import Any


def _filter_clauses(filters: dict[str, str] | None) -> list[dict[str, Any]]:
    return [{"term": {field: value}} for field, value in (filters or {}).items()]


def bm25_query(
    text: str,
    *,
    filters: dict[str, str] | None = None,
    size: int = 10,
) -> dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": text, "fields": ["text", "heading_path^1.5"]}}
                ],
                "filter": _filter_clauses(filters),
            }
        },
    }


def knn_query(
    vector: list[float],
    *,
    filters: dict[str, str] | None = None,
    size: int = 10,
) -> dict[str, Any]:
    knn: dict[str, Any] = {"embedding": {"vector": vector, "k": size}}
    clauses = _filter_clauses(filters)
    if clauses:
        knn["embedding"]["filter"] = {"bool": {"filter": clauses}}
    return {"size": size, "query": {"knn": knn}}
