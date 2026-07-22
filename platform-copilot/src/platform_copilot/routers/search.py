"""POST /search — hybrid retrieval, no generation."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from platform_copilot.dependencies import get_retriever
from platform_copilot.schemas.api import QueryRequest, SearchHit, SearchResponse
from platform_copilot.services.retrieval.retriever import HybridRetriever

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> SearchResponse:
    chunks = retriever.retrieve(request.question, k=request.k, filters=request.filters or None)
    return SearchResponse(
        hits=[
            SearchHit(chunk_id=chunk.id, heading_path=chunk.heading_path, text=chunk.text)
            for chunk in chunks
        ]
    )
