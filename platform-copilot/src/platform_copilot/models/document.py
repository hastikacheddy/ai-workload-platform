"""Document metadata — the source of truth. OpenSearch is a derived index."""

from __future__ import annotations

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from platform_copilot.models.base import Base


class DocumentRow(Base):
    __tablename__ = "documents"

    slug: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String, index=True)
    source_ref: Mapped[str] = mapped_column(String)
    doc_metadata: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    text: Mapped[str] = mapped_column(Text)
