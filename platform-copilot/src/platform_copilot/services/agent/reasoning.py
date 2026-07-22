"""LLM-backed decisions the agent makes: scope guardrail, relevance grading, rewrite.

Each is a small, single-purpose prompt with a parseable answer. Keeping them here
(not inline in the graph) makes the graph readable and these decisions unit-testable.
"""

from __future__ import annotations

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.llm.base import LLM


def _is_yes(text: str) -> bool:
    return text.strip().lower().startswith("y")


def classify_in_scope(llm: LLM, question: str) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify whether the question is about a software platform or its "
                "operations (runbooks, incidents, services, the ML platform). "
                "Answer strictly 'yes' or 'no'."
            ),
        },
        {"role": "user", "content": question},
    ]
    return _is_yes(llm.generate(messages))


def grade_relevance(llm: LLM, question: str, chunks: list[Chunk]) -> bool:
    context = "\n\n".join(chunk.text for chunk in chunks) or "(no context)"
    messages = [
        {
            "role": "system",
            "content": (
                "Does the CONTEXT contain information that answers the QUESTION? "
                "Answer strictly 'yes' or 'no'."
            ),
        },
        {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}"},
    ]
    return _is_yes(llm.generate(messages))


def rewrite_query(llm: LLM, question: str, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the search query to improve retrieval for the question. "
                "Return only the rewritten query."
            ),
        },
        {"role": "user", "content": f"Question: {question}\nCurrent query: {query}"},
    ]
    return llm.generate(messages).strip() or query
