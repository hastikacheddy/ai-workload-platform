from platform_copilot.schemas.document import ParsedDocument, Section
from platform_copilot.services.chunking import _split, chunk_document


def _doc() -> ParsedDocument:
    long_text = " ".join(f"sentence number {i}." for i in range(80))
    return ParsedDocument(
        slug="d",
        title="Doc",
        source_type="runbook",
        source_ref="x.md",
        sections=[
            Section(heading="Short", level=2, text="tiny."),
            Section(heading="Long", level=2, text=long_text),
        ],
    )


def test_chunks_respect_max_size_and_have_unique_ids() -> None:
    chunks = chunk_document(_doc(), max_chars=200, overlap=40)
    assert len(chunks) >= 3  # short -> 1 chunk, long -> several
    assert all(len(c.text) <= 200 for c in chunks)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert any(c.heading_path == "Doc > Short" for c in chunks)
    assert any(c.heading_path == "Doc > Long" for c in chunks)


def test_split_produces_deterministic_overlap() -> None:
    text = "".join(chr(ord("a") + (i % 26)) for i in range(500))  # no natural breaks
    parts = _split(text, max_chars=200, overlap=50)
    assert len(parts) == 3
    assert all(len(p) <= 200 for p in parts)
    assert parts[1][:50] == parts[0][-50:]  # 50-char overlap carried forward
