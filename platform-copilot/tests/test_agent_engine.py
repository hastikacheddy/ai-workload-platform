from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.agent.graph import AgentPipeline
from platform_copilot.services.embeddings.fake import HashEmbedder
from platform_copilot.services.llm.fake import ScriptedLLM
from platform_copilot.services.rag.pipeline import AnswerEngine
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever


def _retriever() -> HybridRetriever:
    chunks = [
        Chunk(doc_slug="r", ordinal=i, heading_path=f"Runbook > S{i}", text=f"drift signal {i} psi")
        for i in range(4)
    ]
    embedder = HashEmbedder(dim=64)
    return HybridRetriever(InMemoryBackend(chunks, embedder), embedder)


def test_agent_pipeline_matches_answer_engine_signature() -> None:
    """The agent must be drop-in swappable for RagPipeline behind /ask."""
    llm = ScriptedLLM(
        {
            "You are a classifier": ["yes"],
            "You are a grader": ["yes"],
            "Platform Copilot": ["grounded answer [1]"],
        }
    )
    engine: AnswerEngine = AgentPipeline(_retriever(), llm)

    result = engine.answer("what indicates drift?", k=2, filters=None)

    assert result.answer == "grounded answer [1]"
    assert len(result.chunks) <= 2  # per-request k is threaded into the graph state
