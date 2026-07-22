"""The RAG pipeline: retrieve -> build grounded prompt -> generate -> attach citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.llm.base import LLM
from platform_copilot.services.rag.prompt import build_messages, citations
from platform_copilot.services.retrieval.retriever import HybridRetriever


@dataclass
class Answer:
    answer: str
    citations: list[dict[str, str]]
    chunks: list[Chunk]


class AnswerEngine(Protocol):
    """Anything that answers a question — the plain pipeline or the cache wrapper."""

    def answer(
        self, question: str, *, k: int = 5, filters: dict[str, str] | None = None
    ) -> Answer: ...


class RagPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM) -> None:
        self._retriever = retriever
        self._llm = llm

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> Answer:
        chunks = self._retriever.retrieve(question, k=k, filters=filters)
        messages = build_messages(question, chunks)
        text = self._llm.generate(messages)
        return Answer(answer=text, citations=citations(chunks), chunks=chunks)
