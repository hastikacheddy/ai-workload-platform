from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.embeddings.fake import HashEmbedder
from platform_copilot.services.llm.fake import StubLLM
from platform_copilot.services.rag.pipeline import RagPipeline
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever


def test_pipeline_answers_with_citations() -> None:
    chunks = [
        Chunk(doc_slug="r", ordinal=0, heading_path="Runbook > Symptoms", text="PSI above 0.2 indicates drift."),
        Chunk(doc_slug="r", ordinal=1, heading_path="Runbook > Steps", text="Check the feature materialization job."),
    ]
    embedder = HashEmbedder(dim=64)
    retriever = HybridRetriever(InMemoryBackend(chunks, embedder), embedder)
    pipeline = RagPipeline(retriever, StubLLM())

    result = pipeline.answer("what indicates drift?", k=2)

    assert result.answer  # non-empty generated text
    assert result.chunks[0].id == "r::0"
    assert result.citations[0]["chunk_id"] == "r::0"
    assert result.citations[0]["n"] == "1"
