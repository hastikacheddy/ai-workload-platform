"""Stable cache keys.

The key is a hash of the normalized question plus the parameters that change the
answer (k, filters), so semantically identical requests collide and hit the cache.
"""

from __future__ import annotations

import hashlib
import json


def answer_key(question: str, *, k: int, filters: dict[str, str] | None) -> str:
    payload = json.dumps(
        {"q": question.strip().lower(), "k": k, "filters": filters or {}},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"ask:{digest}"
