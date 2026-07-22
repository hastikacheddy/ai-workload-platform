from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.agent.graph import REFUSAL, AgentPipeline
from platform_copilot.services.embeddings.fake import HashEmbedder
from platform_copilot.services.llm.fake import ScriptedLLM
from platform_copilot.services.retrieval.memory_backend import InMemoryBackend
from platform_copilot.services.retrieval.retriever import HybridRetriever


def _retriever() -> HybridRetriever:
    chunks = [
        Chunk(doc_slug="r", ordinal=0, heading_path="Runbook > Symptoms", text="PSI above 0.2 indicates drift."),
        Chunk(doc_slug="r", ordinal=1, heading_path="Runbook > Steps", text="Check the feature materialization job."),
    ]
    embedder = HashEmbedder(dim=64)
    return HybridRetriever(InMemoryBackend(chunks, embedder), embedder)


def test_happy_path_generates_grounded_answer() -> None:
    llm = ScriptedLLM(
        {
            "Classify whether": ["yes"],  # in scope
            "Does the CONTEXT": ["yes"],  # relevant
            "Platform Copilot": ["Grounded answer citing [1]."],  # generation
        }
    )
    result = AgentPipeline(_retriever(), llm).answer("what indicates drift?")
    assert result.answer == "Grounded answer citing [1]."
    assert result.citations  # sources attached


def test_guardrail_rejects_out_of_scope() -> None:
    llm = ScriptedLLM({"Classify whether": ["no"]})
    result = AgentPipeline(_retriever(), llm).answer("what's the weather tomorrow?")
    assert result.answer == REFUSAL
    assert result.citations == []


def test_low_relevance_triggers_rewrite_then_answers() -> None:
    llm = ScriptedLLM(
        {
            "Classify whether": ["yes"],
            "Does the CONTEXT": ["no", "yes"],  # miss, then hit after rewrite
            "Rewrite the search query": ["psi drift feature materialization"],
            "Platform Copilot": ["Answer after rewrite [1]."],
        }
    )
    agent = AgentPipeline(_retriever(), llm, max_attempts=2)

    result = agent.answer("drift?")

    assert result.answer == "Answer after rewrite [1]."
    assert any("Rewrite the search query" in call for call in llm.calls)  # rewrite happened
