"""POST /ask — full RAG: retrieve, generate a grounded answer, return citations."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from platform_copilot.dependencies import get_pipeline
from platform_copilot.schemas.api import AskResponse, Citation, QueryRequest
from platform_copilot.services.rag.pipeline import AnswerEngine

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: QueryRequest,
    pipeline: AnswerEngine = Depends(get_pipeline),
) -> AskResponse:
    result = pipeline.answer(request.question, k=request.k, filters=request.filters or None)
    return AskResponse(
        answer=result.answer,
        citations=[Citation(**citation) for citation in result.citations],
    )
