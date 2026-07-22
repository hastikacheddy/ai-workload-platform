"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    filters: dict[str, str] = Field(default_factory=dict)


class SearchHit(BaseModel):
    chunk_id: str
    heading_path: str
    text: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class Citation(BaseModel):
    n: str
    source: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
