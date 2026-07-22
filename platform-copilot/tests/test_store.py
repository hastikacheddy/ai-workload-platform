from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from platform_copilot.models.base import Base
from platform_copilot.services.chunking import chunk_document
from platform_copilot.services.ingestion.markdown import parse_markdown
from platform_copilot.services.store import DocumentStore

MARKDOWN = "---\ntitle: R\nseverity: high\n---\n# R\n\n## A\n\ntext a\n\n## B\n\ntext b\n"


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_upsert_persists_document_and_chunks(session: Session) -> None:
    doc = parse_markdown(MARKDOWN, source_type="runbook", source_ref="r.md")
    chunks = chunk_document(doc, max_chars=100, overlap=10)
    store = DocumentStore(session)

    store.upsert_document(doc, chunks)

    assert store.count_documents() == 1
    saved = store.get_document("r")
    assert saved is not None
    assert saved.doc_metadata["severity"] == "high"
    assert len(store.chunks_for("r")) == len(chunks)


def test_reingest_is_idempotent(session: Session) -> None:
    doc = parse_markdown(MARKDOWN, source_type="runbook", source_ref="r.md")
    chunks = chunk_document(doc, max_chars=100, overlap=10)
    store = DocumentStore(session)

    store.upsert_document(doc, chunks)
    store.upsert_document(doc, chunks)  # re-run the DAG

    assert store.count_documents() == 1
    assert len(store.chunks_for("r")) == len(chunks)  # no duplicate chunk rows
