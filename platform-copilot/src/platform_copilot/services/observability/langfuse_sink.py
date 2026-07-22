"""Langfuse TraceSink.

INTEGRATION-ONLY: needs LANGFUSE_* keys and network, so it runs when the stack is
configured (M5), not in the offline tests. Implements the TraceSink Protocol.
"""

from __future__ import annotations

from platform_copilot.services.observability.base import QueryTrace


class LangfuseSink:
    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    def record(self, trace: QueryTrace) -> None:
        self._client.trace(
            name="ask",
            input=trace.question,
            metadata={
                "latency_ms": trace.latency_ms,
                "num_chunks": trace.num_chunks,
                "answer_chars": trace.answer_chars,
            },
        )
