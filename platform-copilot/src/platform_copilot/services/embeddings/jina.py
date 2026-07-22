"""Jina AI embeddings client.

INTEGRATION-ONLY: needs JINA_API_KEY and network access. Implements the Embedder
Protocol, so offline dev falls back to HashEmbedder with no code change.
"""

from __future__ import annotations

import httpx


class JinaEmbedder:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jina-embeddings-v3",
        dim: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.dim = dim
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
