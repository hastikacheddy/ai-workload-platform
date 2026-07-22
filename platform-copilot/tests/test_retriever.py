from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.embeddings.fake import HashEmbedder
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            doc_slug="r",
            ordinal=0,
            heading_path="Runbook > Symptoms",
            text="PSI above 0.2 indicates drift.",
            metadata={"source_type": "runbook", "severity": "high"},
        ),
        Chunk(
            doc_slug="r",
            ordinal=1,
            heading_path="Runbook > Steps",
            text="Check the feature materialization job.",
            metadata={"source_type": "runbook", "severity": "high"},
        ),
        Chunk(
            doc_slug="p",
            ordinal=0,
            heading_path="Postmortem > Cause",
            text="A bad deploy exhausted the database connection pool.",
            metadata={"source_type": "postmortem"},
        ),
    ]


def _retriever() -> HybridRetriever:
    embedder = HashEmbedder(dim=128)
    return HybridRetriever(InMemoryBackend(_chunks(), embedder), embedder)


def test_retrieve_ranks_relevant_chunk_first() -> None:
    hits = _retriever().retrieve("what indicates drift?", k=2)
    assert hits
    assert hits[0].id == "r::0"


def test_filters_restrict_to_source_type() -> None:
    hits = _retriever().retrieve(
        "deploy connection pool", k=5, filters={"source_type": "postmortem"}
    )
    assert hits
    assert all(hit.metadata.get("source_type") == "postmortem" for hit in hits)
    assert any(hit.id == "p::0" for hit in hits)
