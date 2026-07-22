"""Chunk rows — kept in Postgres alongside documents so the OpenSearch index can be
rebuilt from the source of truth at any time."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from platform_copilot.models.base import Base


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "{doc_slug}::{ordinal}"
    doc_slug: Mapped[str] = mapped_column(String, ForeignKey("documents.slug"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
