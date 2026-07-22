"""Normalized representation of an ingested document.

Every source (Markdown ADR/runbook, HTML postmortem, PDF via Docling later) is
parsed into this shape before it is stored in Postgres and indexed in OpenSearch.
Keeping one shape means the retrieval and RAG layers never care where a doc came from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A heading-delimited slice of a document — the unit we later chunk and cite."""

    heading: str
    level: int
    text: str


class ParsedDocument(BaseModel):
    slug: str
    title: str
    source_type: str  # e.g. "adr" | "runbook" | "model_card" | "postmortem"
    source_ref: str  # file path or URL the document came from
    metadata: dict[str, str] = Field(default_factory=dict)
    sections: list[Section] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Full plain-text rendering, used for indexing and eval."""
        return "\n\n".join(
            f"{section.heading}\n{section.text}".strip() for section in self.sections
        ).strip()
