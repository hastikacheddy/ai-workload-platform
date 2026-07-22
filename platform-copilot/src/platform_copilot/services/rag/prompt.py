"""Build the grounded prompt sent to the LLM.

The system prompt is the first line of hallucination defense: answer only from the
numbered context, cite sources as [n], and admit when the answer is not present.
Citations are returned alongside so the API can render clickable sources.
"""

from __future__ import annotations

from platform_copilot.schemas.chunk import Chunk

SYSTEM_PROMPT = (
    "You are the Platform Copilot, an assistant for on-call and platform engineers. "
    "Answer the question using ONLY the numbered context sources below. "
    "Cite the sources you rely on inline as [n]. "
    "If the answer is not in the context, say you don't know and suggest where to look. "
    "Be concise and operational."
)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{n}] ({chunk.heading_path})\n{chunk.text}" for n, chunk in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    user = f"Question: {question}\n\nContext sources:\n{format_context(chunks)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def citations(chunks: list[Chunk]) -> list[dict[str, str]]:
    """Map each context number back to its source, for the response payload."""
    return [
        {"n": str(n), "source": chunk.heading_path, "chunk_id": chunk.id}
        for n, chunk in enumerate(chunks, start=1)
    ]
