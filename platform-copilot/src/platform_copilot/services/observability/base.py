"""The observability boundary — a Langfuse sink in production, a fake in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class QueryTrace:
    question: str
    latency_ms: float
    num_chunks: int
    answer_chars: int  # cheap proxy for response size / cost


class TraceSink(Protocol):
    def record(self, trace: QueryTrace) -> None: ...
