"""Parse HTML documents (e.g. public postmortems) into ParsedDocument.

Headings (h1–h4) delimit sections; script/style/nav/footer are dropped. Same
output shape as the Markdown parser, so downstream code treats every source alike.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from platform_copilot.schemas.document import ParsedDocument, Section

_HEADINGS = ("h1", "h2", "h3", "h4")
_BLOCKS = ["h1", "h2", "h3", "h4", "p", "li", "pre"]


def parse_html(
    html: str,
    *,
    source_type: str,
    source_ref: str,
    slug: str | None = None,
) -> ParsedDocument:
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "nav", "footer"]):
        junk.decompose()

    body = soup.body or soup
    sections: list[Section] = []
    heading, level, buffer = "Preamble", 0, []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content or heading != "Preamble":
            sections.append(Section(heading=heading, level=level, text=content))

    for element in body.find_all(_BLOCKS):
        text = element.get_text(" ", strip=True)
        if element.name in _HEADINGS:
            flush()
            level, heading, buffer = int(element.name[1]), text, []
        elif text:
            buffer.append(text)
    flush()

    resolved_slug = slug or Path(source_ref).stem
    title_tag = soup.find("h1") or soup.title
    first_h1 = sections[0].heading if sections and sections[0].level == 1 else ""
    title = (title_tag.get_text(strip=True) if title_tag else "") or first_h1 or resolved_slug
    return ParsedDocument(
        slug=resolved_slug,
        title=title,
        source_type=source_type,
        source_ref=source_ref,
        metadata={},
        sections=sections,
    )
