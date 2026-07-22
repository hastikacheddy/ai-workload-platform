from pathlib import Path

from platform_copilot.services.ingestion.markdown import parse_markdown

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def test_parse_frontmatter_and_sections() -> None:
    doc = parse_markdown(
        (CORPUS / "runbook-drift-alert.md").read_text(encoding="utf-8"),
        source_type="runbook",
        source_ref="corpus/runbook-drift-alert.md",
    )

    assert doc.slug == "runbook-drift-alert"
    assert doc.title == "Runbook — Demand Forecaster Drift Alert"
    assert doc.metadata["severity"] == "high"
    assert doc.metadata["service"] == "demand-forecaster"

    headings = [section.heading for section in doc.sections]
    assert "Symptoms" in headings
    assert "First steps" in headings
    assert "Escalation" in headings

    assert "PSI" in doc.text
    assert "promotion gate" in doc.text


def test_plain_markdown_without_frontmatter() -> None:
    doc = parse_markdown(
        "# Title\n\nBody paragraph.\n\n## Details\n\nMore text.",
        source_type="adr",
        source_ref="notes/example.md",
    )

    assert doc.title == "Title"
    assert doc.metadata == {}
    assert [s.heading for s in doc.sections] == ["Title", "Details"]
