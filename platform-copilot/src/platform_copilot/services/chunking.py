"""Section-aware chunking.

Chunks never cross a section boundary, so every chunk keeps a clean heading path
for citation. Long sections are split with character overlap, preferring paragraph /
line / sentence boundaries so retrieval units stay readable.
"""

from __future__ import annotations

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.schemas.document import ParsedDocument


def chunk_document(
    doc: ParsedDocument,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for section in doc.sections:
        heading_path = (
            doc.title if section.heading == "Preamble" else f"{doc.title} > {section.heading}"
        )
        for piece in _split(section.text, max_chars, overlap):
            chunks.append(
                Chunk(
                    doc_slug=doc.slug,
                    ordinal=ordinal,
                    heading_path=heading_path,
                    text=piece,
                    metadata=doc.metadata,
                )
            )
            ordinal += 1
    return chunks


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:  # try to end on a natural boundary before the hard limit
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = text.rfind("\n", start, end)
            if boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= length:
            break
        start = max(end - overlap, start + 1)  # always makes progress
    return windows
