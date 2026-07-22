"""Reciprocal Rank Fusion — how keyword and vector results become one ranking.

RRF combines ranked lists using only rank position, so BM25 scores and cosine
similarities (which are not comparable) can be merged without tuning weights.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list of (id, score), best first.

    ``k`` dampens the contribution of lower ranks; 60 is the common default.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
