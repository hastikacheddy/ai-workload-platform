"""Persistence for ingested documents and their chunks.

``upsert_document`` is idempotent: re-ingesting the same document replaces its rows
instead of duplicating them, which is what makes the Airflow ingestion DAG safe to
re-run. Backed by SQLAlchemy, so the same code runs on SQLite (tests) or Postgres.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from platform_copilot.models.chunk import ChunkRow
from platform_copilot.models.document import DocumentRow
from platform_copilot.schemas.chunk import Chunk
from platform_copilot.schemas.document import ParsedDocument


class DocumentStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_document(self, doc: ParsedDocument, chunks: list[Chunk]) -> None:
        self._session.merge(
            DocumentRow(
                slug=doc.slug,
                title=doc.title,
                source_type=doc.source_type,
                source_ref=doc.source_ref,
                doc_metadata=doc.metadata,
                text=doc.text,
            )
        )
        # Replace chunks so a re-ingest never leaves stale rows behind.
        self._session.execute(delete(ChunkRow).where(ChunkRow.doc_slug == doc.slug))
        for chunk in chunks:
            self._session.add(
                ChunkRow(
                    id=chunk.id,
                    doc_slug=chunk.doc_slug,
                    ordinal=chunk.ordinal,
                    heading_path=chunk.heading_path,
                    text=chunk.text,
                )
            )
        self._session.commit()

    def get_document(self, slug: str) -> DocumentRow | None:
        return self._session.get(DocumentRow, slug)

    def chunks_for(self, slug: str) -> list[ChunkRow]:
        stmt = select(ChunkRow).where(ChunkRow.doc_slug == slug).order_by(ChunkRow.ordinal)
        return list(self._session.scalars(stmt))

    def count_documents(self) -> int:
        return self._session.scalar(select(func.count()).select_from(DocumentRow)) or 0
