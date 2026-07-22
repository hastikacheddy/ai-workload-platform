from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.rag.prompt import build_messages, citations


def _chunks() -> list[Chunk]:
    return [
        Chunk(doc_slug="r", ordinal=0, heading_path="Runbook > Symptoms", text="PSI above 0.2."),
        Chunk(doc_slug="r", ordinal=1, heading_path="Runbook > First steps", text="Check the dashboard."),
    ]


def test_build_messages_is_grounded_and_numbered() -> None:
    messages = build_messages("what indicates drift?", _chunks())

    assert messages[0]["role"] == "system"
    assert "ONLY" in messages[0]["content"]

    user = messages[1]["content"]
    assert "what indicates drift?" in user
    assert "[1]" in user and "[2]" in user
    assert "PSI above 0.2." in user


def test_citations_map_numbers_to_sources() -> None:
    cites = citations(_chunks())
    assert cites[0] == {"n": "1", "source": "Runbook > Symptoms", "chunk_id": "r::0"}
    assert cites[1]["chunk_id"] == "r::1"
