"""Liveness endpoint. Deep readiness checks (Postgres, OpenSearch) arrive in M1."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
