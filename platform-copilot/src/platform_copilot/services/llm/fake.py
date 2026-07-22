"""Deterministic LLM stubs for tests — no model, no network."""

from __future__ import annotations


class StubLLM:
    """Returns one fixed grounded-looking answer. For the simple RAG pipeline test."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        return "Based on the retrieved context, the key signal is described in source [1]."


class ScriptedLLM:
    """Returns canned responses keyed by a substring of the prompt.

    Each key maps to a queue of responses (popped in order), so a multi-step agent
    flow — grade "no", rewrite, then grade "yes" — can be scripted deterministically.
    Records every prompt in ``calls`` for assertions.
    """

    def __init__(self, rules: dict[str, list[str]], *, default: str = "yes") -> None:
        self._rules = {key: list(values) for key, values in rules.items()}
        self._default = default
        self.calls: list[str] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = " ".join(message["content"] for message in messages)
        self.calls.append(text)
        for key, responses in self._rules.items():
            if key.lower() in text.lower() and responses:
                return responses.pop(0)
        return self._default
