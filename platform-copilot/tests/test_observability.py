from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.observability.memory import InMemorySink
from platform_copilot.services.rag.observed import ObservedPipeline
from platform_copilot.services.rag.pipeline import Answer


class FakeInner:
    def answer(self, question: str, *, k: int = 5, filters: dict[str, str] | None = None) -> Answer:
        return Answer(
            answer="hello",
            citations=[],
            chunks=[Chunk(doc_slug="r", ordinal=0, heading_path="h", text="t")],
        )


def test_observed_records_a_trace_per_call() -> None:
    sink = InMemorySink()
    pipe = ObservedPipeline(FakeInner(), sink)

    result = pipe.answer("what indicates drift?")

    assert result.answer == "hello"
    assert len(sink.traces) == 1
    trace = sink.traces[0]
    assert trace.question == "what indicates drift?"
    assert trace.num_chunks == 1
    assert trace.answer_chars == len("hello")
    assert trace.latency_ms >= 0
