"""A deterministic, dependency-free embedder for tests and offline development.

It is NOT semantically meaningful — a hashed bag-of-words mapped to a stable unit
vector. Its job is to let the chunking → embed → index → fuse plumbing run and be
tested without downloading a model or calling a network. Swap in a real Embedder
(same interface) for actual retrieval quality.
"""

from __future__ import annotations

import hashlib
import math


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vector[digest % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
