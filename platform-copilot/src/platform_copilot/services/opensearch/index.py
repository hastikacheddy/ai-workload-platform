"""OpenSearch index definition for the chunk store.

One index serves both retrieval arms: an analyzed ``text`` field for BM25 and a
``knn_vector`` field for dense search. Keyword fields back exact filters
(source_type, service, severity).
"""

from __future__ import annotations

from typing import Any


def index_settings(embedding_dim: int = 1024) -> dict[str, Any]:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "doc_slug": {"type": "keyword"},
                "ordinal": {"type": "integer"},
                "heading_path": {"type": "text"},
                "text": {"type": "text", "analyzer": "english"},
                "source_type": {"type": "keyword"},
                "service": {"type": "keyword"},
                "severity": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
