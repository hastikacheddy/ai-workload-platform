"""Non-production TraceSinks: collect in memory (tests) or drop (default offline)."""

from __future__ import annotations

from platform_copilot.services.observability.base import QueryTrace


class InMemorySink:
    def __init__(self) -> None:
        self.traces: list[QueryTrace] = []

    def record(self, trace: QueryTrace) -> None:
        self.traces.append(trace)


class NullSink:
    def record(self, trace: QueryTrace) -> None:
        return None
