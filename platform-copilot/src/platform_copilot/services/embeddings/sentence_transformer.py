"""Embedder backed by a Sentence-Transformers model — real semantic vectors.

A small CPU-friendly model such as ``all-MiniLM-L6-v2`` makes the in-memory demo's
vector arm actually semantic (unlike the HashEmbedder placeholder). Vectors are
L2-normalized, so the in-memory backend's dot product is cosine similarity.
Implements the Embedder Protocol.

Needs the `st` extra:  pip install ".[st]"
"""

from __future__ import annotations


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
