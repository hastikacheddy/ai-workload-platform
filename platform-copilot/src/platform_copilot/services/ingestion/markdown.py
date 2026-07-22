"""Parse Markdown documents (ADRs, runbooks, model cards) into ParsedDocument.

Markdown covers most of the platform corpus, so it is the first parser. PDF/HTML
via Docling arrives when the corpus needs it (kept out of the core dependencies).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from platform_copilot.schemas.document import ParsedDocument, Section

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(
    text: str,
    *,
    source_type: str,
    source_ref: str,
    slug: str | None = None,
) -> ParsedDocument:
    """Split Markdown into an optional YAML frontmatter block + heading sections."""
    metadata: dict[str, str] = {}
    match = _FRONTMATTER.match(text)
    if match:
        loaded: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
        metadata = {str(key): str(value) for key, value in loaded.items()}
        text = text[match.end() :]

    sections: list[Section] = []
    heading, level, buffer = "Preamble", 0, []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content or heading != "Preamble":
            sections.append(Section(heading=heading, level=level, text=content))

    for line in text.splitlines():
        head = _HEADING.match(line)
        if head:
            flush()
            level, heading, buffer = len(head.group(1)), head.group(2).strip(), []
        else:
            buffer.append(line)
    flush()

    first_h1 = sections[0].heading if sections and sections[0].level == 1 else None
    resolved_slug = slug or Path(source_ref).stem
    title = metadata.get("title") or first_h1 or resolved_slug
    return ParsedDocument(
        slug=resolved_slug,
        title=title,
        source_type=source_type,
        source_ref=source_ref,
        metadata=metadata,
        sections=sections,
    )
