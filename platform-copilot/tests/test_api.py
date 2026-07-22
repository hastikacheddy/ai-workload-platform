from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from platform_copilot.dependencies import get_pipeline, get_retriever
from platform_copilot.main import app
from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.embeddings.fake import HashEmbedder
from platform_copilot.services.llm.fake import StubLLM
from platform_copilot.services.rag.pipeline import RagPipeline
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever

CHUNKS = [
    Chunk(
        doc_slug="r",
        ordinal=0,
        heading_path="Runbook > Symptoms",
        text="PSI above 0.2 indicates drift.",
        metadata={"source_type": "runbook"},
    ),
    Chunk(
        doc_slug="r",
        ordinal=1,
        heading_path="Runbook > Steps",
        text="Check the feature materialization job.",
        metadata={"source_type": "runbook"},
    ),
]


def _retriever() -> HybridRetriever:
    embedder = HashEmbedder(dim=64)
    return HybridRetriever(InMemoryBackend(CHUNKS, embedder), embedder)


@pytest.fixture
def client() -> Iterator[TestClient]:
    retriever = _retriever()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(retriever, StubLLM())
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_search_endpoint_returns_hits(client: TestClient) -> None:
    response = client.post("/search", json={"question": "what indicates drift?"})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits
    assert hits[0]["chunk_id"] == "r::0"


def test_ask_endpoint_returns_answer_and_citations(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "what indicates drift?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"][0]["chunk_id"] == "r::0"
