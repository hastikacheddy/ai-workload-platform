from platform_copilot.services.rag.pipeline import Answer
from platform_copilot.services.rag.render import render_answer


def _answer(citations: list[dict[str, str]]) -> Answer:
    return Answer(answer="Do X then Y.", citations=citations, chunks=[])


def test_appends_sources() -> None:
    out = render_answer(
        _answer([{"n": "1", "source": "Runbook > First steps", "chunk_id": "r::3"}])
    )
    assert "Do X then Y." in out
    assert "[1] Runbook > First steps" in out


def test_plain_when_no_citations() -> None:
    assert render_answer(_answer([])) == "Do X then Y."


def test_caps_source_list() -> None:
    citations = [{"n": str(i), "source": f"S{i}", "chunk_id": f"c::{i}"} for i in range(1, 9)]
    out = render_answer(_answer(citations), max_sources=3)
    assert "[3] S3" in out
    assert "[4] S4" not in out
