from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.agent.graph import REFUSAL, AgentPipeline
from platform_copilot.services.agent.reasoning import parse_yes_no
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
            "You are a classifier": ["yes"],  # in scope
            "You are a grader": ["yes"],  # relevant
            "Platform Copilot": ["Grounded answer citing [1]."],  # generation
        }
    )
    result = AgentPipeline(_retriever(), llm).answer("what indicates drift?")
    assert result.answer == "Grounded answer citing [1]."
    assert result.citations  # sources attached


def test_refuses_only_when_classifier_and_retrieval_agree() -> None:
    """Out of scope = the classifier says no AND retrieval found nothing relevant."""
    llm = ScriptedLLM({"You are a classifier": ["no"], "You are a grader": ["no"]})
    result = AgentPipeline(_retriever(), llm).answer("what's the weather tomorrow?")
    assert result.answer == REFUSAL
    assert result.citations == []


def test_weak_classifier_is_overridden_by_relevant_retrieval() -> None:
    """A classifier that misreads domain jargon must not refuse an answerable question.

    Regression for a real live failure: llama3.2:1b classified "the promotion gate
    keeps rejecting candidates" as out of scope even with that exact few-shot
    example. Retrieval corroboration rescues it.
    """
    llm = ScriptedLLM(
        {
            "You are a classifier": ["no"],  # wrong verdict
            "You are a grader": ["yes"],  # but the corpus does answer it
            "Platform Copilot": ["Grounded answer [1]."],
        }
    )

    result = AgentPipeline(_retriever(), llm).answer("promotion gate keeps rejecting candidates")

    assert result.answer == "Grounded answer [1]."
    assert result.citations


def test_low_relevance_triggers_rewrite_then_answers() -> None:
    llm = ScriptedLLM(
        {
            "You are a classifier": ["yes"],
            "You are a grader": ["no", "yes"],  # miss, then hit after rewrite
            "Rewrite the search query": ["psi drift feature materialization"],
            "Platform Copilot": ["Answer after rewrite [1]."],
        }
    )
    agent = AgentPipeline(_retriever(), llm, max_attempts=2)

    result = agent.answer("drift?")

    assert result.answer == "Answer after rewrite [1]."
    assert any("Rewrite the search query" in call for call in llm.calls)  # rewrite happened


def test_unparseable_guardrail_reply_fails_open() -> None:
    """A model that ignores the yes/no format must not cause a false refusal.

    Regression test for a real live failure: llama3.2:1b answered the guardrail
    prompt with "I can't help with that.", which a startswith('y') check read as
    "no" and refused a valid operational question.
    """
    llm = ScriptedLLM(
        {
            "You are a classifier": ["I can't help with that."],  # not a decision
            "You are a grader": ["yes"],
            "Platform Copilot": ["Grounded answer [1]."],
        }
    )

    result = AgentPipeline(_retriever(), llm).answer("how do I escalate a drift alert?")

    assert result.answer == "Grounded answer [1]."  # answered, not refused
    assert result.citations


def test_parse_yes_no_is_tri_state() -> None:
    assert parse_yes_no("yes") is True
    assert parse_yes_no("No.") is False
    assert parse_yes_no("**yes**") is True
    assert parse_yes_no("Yes, that is a platform question") is True
    assert parse_yes_no("I can't help with that.") is None  # undecidable -> caller fails open
    # Regression: a substring check found "no" inside "cannot" and refused wrongly.
    assert parse_yes_no("I cannot provide a forecast.") is None
    assert parse_yes_no("I do not know, but that is a deployment topic") is None
