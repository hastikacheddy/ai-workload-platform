"""Ingest corpus/ into Postgres (source of truth) + OpenSearch (index).

This is the batch job the Airflow DAG will call. Idempotent: re-running replaces a
document's rows rather than duplicating them.
"""

from __future__ import annotations

from pathlib import Path

from opensearchpy import OpenSearch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from platform_copilot.config import get_settings
from platform_copilot.dependencies import _build_embedder
from platform_copilot.models.base import Base
from platform_copilot.services.chunking import chunk_document
from platform_copilot.services.ingestion.html import parse_html
from platform_copilot.services.ingestion.markdown import parse_markdown
from platform_copilot.services.opensearch.backend import OpenSearchBackend
from platform_copilot.services.store import DocumentStore

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.postgres_dsn)
    Base.metadata.create_all(engine)

    client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}]
    )
    backend = OpenSearchBackend(client, settings.opensearch_index)
    embedder = _build_embedder()
    backend.ensure_index(embedder.dim)

    documents = []
    for path in sorted(CORPUS.glob("*.md")):
        documents.append(
            parse_markdown(path.read_text(encoding="utf-8"), source_type="runbook", source_ref=path.name)
        )
    for path in sorted(CORPUS.glob("*.html")):
        documents.append(
            parse_html(path.read_text(encoding="utf-8"), source_type="postmortem", source_ref=path.name)
        )

    total = 0
    with Session(engine) as session:
        store = DocumentStore(session)
        for doc in documents:
            chunks = chunk_document(doc)
            store.upsert_document(doc, chunks)
            backend.index_chunks(chunks, embedder.embed([chunk.text for chunk in chunks]))
            total += len(chunks)
            print(f"  {doc.slug}: {len(chunks)} chunks")

    client.indices.refresh(index=settings.opensearch_index)
    print(f"ingested {len(documents)} docs / {total} chunks into Postgres + OpenSearch")


if __name__ == "__main__":
    main()
