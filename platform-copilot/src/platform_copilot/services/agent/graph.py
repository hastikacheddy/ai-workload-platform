"""The agentic RAG loop as a LangGraph state machine.

    START -> guardrail -> retrieve -> grade -> (generate | rewrite | reject)
    rewrite -> retrieve            (bounded by max_attempts)
    generate / reject -> END

Scope is *corroborated*, not trusted: the classifier's verdict is carried in state
and only acted on after retrieval, so a weak classifier that misreads domain jargon
cannot refuse a question the corpus can actually answer.

The agent only adds control flow a static chain can't express: refusing out-of-scope
questions, detecting low-relevance retrievals, and rewriting + retrying. Every
decision goes through the injected LLM, so the whole graph is testable with a fake.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.agent.reasoning import (
    classify_in_scope,
    grade_relevance,
    rewrite_query,
)
from platform_copilot.services.llm.base import LLM
from platform_copilot.services.rag.pipeline import Answer
from platform_copilot.services.rag.prompt import build_messages, citations
from platform_copilot.services.retrieval.retriever import HybridRetriever

REFUSAL = "I can only help with platform and operations questions."


class AgentState(TypedDict, total=False):
    question: str
    query: str
    attempts: int
    chunks: list[Chunk]
    relevant: bool
    in_scope: bool
    answer: str
    citations: list[dict[str, str]]
    k: int
    filters: dict[str, str] | None


def build_agent(
    retriever: HybridRetriever,
    llm: LLM,
    *,
    max_attempts: int = 2,
    k: int = 5,
):  # returns a compiled LangGraph
    def guardrail(state: AgentState) -> AgentState:
        return {
            "in_scope": classify_in_scope(llm, state["question"]),
            "query": state["question"],
            "attempts": 0,
        }

    def reject(state: AgentState) -> AgentState:
        return {"answer": REFUSAL, "citations": [], "chunks": []}

    def retrieve(state: AgentState) -> AgentState:
        return {
            "chunks": retriever.retrieve(
                state["query"], k=state.get("k") or k, filters=state.get("filters")
            )
        }

    def grade(state: AgentState) -> AgentState:
        return {"relevant": grade_relevance(llm, state["question"], state.get("chunks", []))}

    def rewrite(state: AgentState) -> AgentState:
        return {
            "query": rewrite_query(llm, state["question"], state["query"]),
            "attempts": state.get("attempts", 0) + 1,
        }

    def generate(state: AgentState) -> AgentState:
        chunks = state.get("chunks", [])
        text = llm.generate(build_messages(state["question"], chunks))
        return {"answer": text, "citations": citations(chunks)}

    def after_grade(state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        # Refuse only when BOTH signals agree the question is off-topic: the
        # classifier said no AND retrieval surfaced nothing relevant. This stops a
        # weak classifier from refusing valid questions the corpus can answer.
        if not state.get("in_scope", True):
            return "reject"
        if state.get("attempts", 0) < max_attempts:
            return "rewrite"
        return "generate"  # retries exhausted: answer with best effort (prompt admits gaps)

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("reject", reject)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "guardrail")
    graph.add_edge("guardrail", "retrieve")  # scope is corroborated after retrieval
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite", "reject": "reject"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("reject", END)
    graph.add_edge("generate", END)
    return graph.compile()


class AgentPipeline:
    """Same ``answer()`` shape as RagPipeline, so it can back /ask once the LLM is live."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        *,
        max_attempts: int = 2,
        k: int = 5,
    ) -> None:
        self._graph = build_agent(retriever, llm, max_attempts=max_attempts, k=k)

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> Answer:
        final = self._graph.invoke({"question": question, "k": k, "filters": filters})
        return Answer(
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            chunks=final.get("chunks", []),
        )
