"""Observability wrapper around any AnswerEngine.

Times each call and emits a QueryTrace (latency, retrieval size, response size) to a
sink. Layered outside the cache (`Observed -> Cached -> RAG`) so a cache hit shows up
as a genuinely fast trace — the data behind a cost/latency dashboard.
"""

from __future__ import annotations

import time

from platform_copilot.services.observability.base import QueryTrace, TraceSink
from platform_copilot.services.rag.pipeline import Answer, AnswerEngine


class ObservedPipeline:
    def __init__(self, inner: AnswerEngine, sink: TraceSink) -> None:
        self._inner = inner
        self._sink = sink

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> Answer:
        start = time.perf_counter()
        result = self._inner.answer(question, k=k, filters=filters)
        latency_ms = (time.perf_counter() - start) * 1000
        self._sink.record(
            QueryTrace(
                question=question,
                latency_ms=latency_ms,
                num_chunks=len(result.chunks),
                answer_chars=len(result.answer),
            )
        )
        return result
