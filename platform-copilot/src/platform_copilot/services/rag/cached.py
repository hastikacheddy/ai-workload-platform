"""Cache wrapper around any AnswerEngine.

On a cache hit the LLM and retrieval are skipped entirely — the big win, since
generation dominates latency and cost. Answers are stored as JSON so chunks and
citations survive the round trip.
"""

from __future__ import annotations

import json

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.cache.base import Cache
from platform_copilot.services.cache.keys import answer_key
from platform_copilot.services.rag.pipeline import Answer, AnswerEngine


class CachedPipeline:
    def __init__(self, inner: AnswerEngine, cache: Cache, *, ttl_seconds: int = 3600) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> Answer:
        key = answer_key(question, k=k, filters=filters)
        cached = self._cache.get(key)
        if cached is not None:
            return _deserialize(cached)
        result = self._inner.answer(question, k=k, filters=filters)
        self._cache.set(key, _serialize(result), ttl_seconds=self._ttl)
        return result


def _serialize(answer: Answer) -> str:
    return json.dumps(
        {
            "answer": answer.answer,
            "citations": answer.citations,
            "chunks": [chunk.model_dump() for chunk in answer.chunks],
        }
    )


def _deserialize(raw: str) -> Answer:
    data = json.loads(raw)
    return Answer(
        answer=data["answer"],
        citations=data["citations"],
        chunks=[Chunk(**chunk) for chunk in data["chunks"]],
    )
