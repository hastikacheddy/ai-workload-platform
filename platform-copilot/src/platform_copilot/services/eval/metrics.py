"""Retrieval quality metrics.

These turn "the hybrid search feels better" into numbers, so every enhancement
(vectors, reranking, chunk-size changes) is judged against a labeled set instead
of vibes. Pure functions — run them against any ranking, live or offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = set(ranked_ids[:k]) & relevant_ids
    return len(hits) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hits / len(top)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant_ids: set[str]


def evaluate(results: dict[str, list[str]], cases: Sequence[EvalCase], k: int = 5) -> dict[str, float]:
    """Aggregate recall@k and MRR over a labeled query set.

    ``results`` maps each query to the ranked chunk ids the system returned.
    """
    if not cases:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recalls, rrs = [], []
    for case in cases:
        ranked = results.get(case.query, [])
        recalls.append(recall_at_k(ranked, case.relevant_ids, k))
        rrs.append(reciprocal_rank(ranked, case.relevant_ids))
    return {
        "recall_at_k": sum(recalls) / len(cases),
        "mrr": sum(rrs) / len(cases),
    }
