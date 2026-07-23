"""LLM-backed decisions the agent makes: scope guardrail, relevance grading, rewrite.

Each is a small, single-purpose prompt with a parseable answer. Keeping them here
(not inline in the graph) makes the graph readable and these decisions unit-testable.

**Fail-open by design.** Small instruction-following models frequently ignore an
"answer yes or no" instruction and reply conversationally instead. When a reply
cannot be parsed as a decision we *proceed* rather than refuse:

* a wrongly-refused operational question is far worse than an off-topic one
  slipping through, and
* the grounded answer prompt already makes the model say "I don't know" when the
  retrieved context does not cover the question.

Consequence, stated plainly: the guardrail is only as strong as the classifier
model. A small model may not recognise platform jargon ("promotion gate", "drift
alert") and will misclassify valid questions as out of scope, so the classifier
prompt carries an explicit domain vocabulary. If the classifier still misfires for
your model, run with ``USE_AGENT=false`` — the single-shot RAG path has no gate.
The graph logic itself is verified independently by driving it with a scripted LLM.
"""

from __future__ import annotations

import re

from platform_copilot.schemas.chunk import Chunk
from platform_copilot.services.llm.base import LLM

# Word-boundary matching matters: a naive "no" substring check fires inside
# "cannot", "not", and "know" — which turned a non-answer into a false refusal.
_YES_RE = re.compile(r"\byes\b")
_NO_RE = re.compile(r"\bno\b")

_PLATFORM_TERMS = (
    "deployments, rollbacks, incidents, runbooks, alerts, monitoring, drift, "
    "model registry, promotion gates, feature pipelines, SLOs, Kubernetes, "
    "serving, retraining"
)


def parse_yes_no(text: str) -> bool | None:
    """True/False when the model actually answered; None when it did not."""
    stripped = text.strip().lower().strip("*_`'\"-. \n")
    if stripped == "yes":
        return True
    if stripped == "no":
        return False

    head = stripped[:80]
    has_yes = bool(_YES_RE.search(head))
    has_no = bool(_NO_RE.search(head))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def _decide(llm: LLM, messages: list[dict[str, str]], *, default: bool) -> bool:
    verdict = parse_yes_no(llm.generate(messages))
    return default if verdict is None else verdict


def classify_in_scope(llm: LLM, question: str) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. Decide whether the question concerns a software "
                "platform or its operations.\n"
                f"Platform topics include: {_PLATFORM_TERMS}.\n"
                "Reply with exactly one word: yes or no. Give no explanation.\n"
                "Examples:\n"
                "Q: how do I roll back a bad deploy? -> yes\n"
                "Q: the promotion gate keeps rejecting my model -> yes\n"
                "Q: what should I do about a drift alert? -> yes\n"
                "Q: what is the capital of France? -> no\n"
                "Q: what is the weather tomorrow? -> no"
            ),
        },
        {"role": "user", "content": f"Q: {question} ->"},
    ]
    # Fail open: answer the question rather than wrongly refuse it.
    return _decide(llm, messages, default=True)


def grade_relevance(llm: LLM, question: str, chunks: list[Chunk]) -> bool:
    context = "\n\n".join(chunk.text for chunk in chunks) or "(no context)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a grader. Decide whether the CONTEXT contains information that "
                "helps answer the QUESTION.\n"
                "Reply with exactly one word: yes or no. Give no explanation."
            ),
        },
        {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}"},
    ]
    # Fail open: produce a grounded (possibly "I don't know") answer rather than
    # burn retries rewriting a query we were unable to grade.
    return _decide(llm, messages, default=True)


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
