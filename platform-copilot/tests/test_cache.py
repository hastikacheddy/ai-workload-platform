from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.cache.memory import InMemoryCache
from platform_copilot.services.rag.cached import CachedPipeline
from platform_copilot.services.rag.pipeline import Answer


class CountingPipeline:
    """Stand-in AnswerEngine that counts how often it actually computes."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, *, k: int = 5, filters: dict[str, str] | None = None) -> Answer:
        self.calls += 1
        return Answer(
            answer=f"ans-{self.calls}",
            citations=[{"n": "1", "source": "Runbook > Symptoms", "chunk_id": "r::0"}],
            chunks=[Chunk(doc_slug="r", ordinal=0, heading_path="Runbook > Symptoms", text="PSI above 0.2.")],
        )


def test_cache_hit_avoids_recompute() -> None:
    inner = CountingPipeline()
    cache = InMemoryCache()
    pipe = CachedPipeline(inner, cache)

    first = pipe.answer("what indicates drift?", k=5)
    second = pipe.answer("what indicates drift?", k=5)

    assert inner.calls == 1  # second request served from cache
    assert first.answer == second.answer == "ans-1"
    assert cache.hits == 1
    assert cache.misses == 1


def test_different_query_is_a_miss() -> None:
    inner = CountingPipeline()
    pipe = CachedPipeline(inner, InMemoryCache())

    pipe.answer("question one")
    pipe.answer("question two")

    assert inner.calls == 2


def test_cache_roundtrip_preserves_chunks_and_citations() -> None:
    inner = CountingPipeline()
    pipe = CachedPipeline(inner, InMemoryCache())

    pipe.answer("q")  # miss -> serialize + store
    hit = pipe.answer("q")  # hit -> deserialize

    assert hit.chunks[0].id == "r::0"
    assert hit.citations[0]["chunk_id"] == "r::0"
