"""A retrieval unit: a bounded slice of a document, embedded and indexed."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    doc_slug: str
    ordinal: int  # position within the source document
    heading_path: str  # e.g. "Runbook — Drift Alert > Symptoms" (kept for citations)
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_slug}::{self.ordinal}"
