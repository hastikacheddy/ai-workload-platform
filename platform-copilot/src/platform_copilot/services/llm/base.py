"""The LLM boundary — a local Ollama model, a hosted API, or the test stub all fit."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...
