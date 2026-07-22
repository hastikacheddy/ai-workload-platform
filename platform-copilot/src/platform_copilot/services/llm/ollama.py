"""Local Ollama chat client.

INTEGRATION-ONLY: needs a running Ollama server with the model pulled, so it is
wired up when the stack is live (M4/M5). Implements the LLM Protocol.
"""

from __future__ import annotations

import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, *, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={"model": self._model, "messages": messages, "stream": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        content: str = response.json()["message"]["content"]
        return content
