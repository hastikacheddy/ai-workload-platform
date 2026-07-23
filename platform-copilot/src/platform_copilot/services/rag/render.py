"""Render an Answer for chat surfaces (Gradio, Telegram).

Deliberately free of any UI dependency, so the formatting both interfaces rely on
is unit-tested offline even though the interfaces themselves need a browser/token.
"""

from __future__ import annotations

from platform_copilot.services.rag.pipeline import Answer


def render_answer(answer: Answer, *, max_sources: int = 5) -> str:
    if not answer.citations:
        return answer.answer
    sources = "\n".join(
        f"[{citation['n']}] {citation['source']}" for citation in answer.citations[:max_sources]
    )
    return f"{answer.answer}\n\n---\nSources:\n{sources}"
