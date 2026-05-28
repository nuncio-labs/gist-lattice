from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MemoryAnalysis(BaseModel):
    gist: str
    valence: float = Field(ge=-1.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    structural_location: str | None = None
    core_project: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_nulls(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        normalized = dict(values)
        for field in ("structural_location", "core_project"):
            value = normalized.get(field)
            if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
                normalized[field] = None
        return normalized


class ExtractedMemory(BaseModel):
    tenant_id: str
    user_id: str
    interaction_id: str
    gist: str
    valence: float
    importance: float
    embedding: list[float]
    entities: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)


class MemoryGist(BaseModel):
    gist: str
    valence: float
    importance: float
    score: float = 0.0
    raw_text: str | None = None
    last_accessed: datetime | None = None


class SemanticContextItem(BaseModel):
    relationship: str
    value: str


class ConsolidationJob(BaseModel):
    job_id: str
    interaction_id: str
    tenant_id: str
    user_id: str
    prompt: str
    response: str
    request_id: str
    attempt: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryDocument(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalResult(BaseModel):
    query: str
    tenant_id: str
    user_id: str
    documents: list[MemoryDocument] = Field(default_factory=list)
    hydrated_context: str = ""
    memory_hits: int = 0


class Principal(BaseModel):
    subject: str
    tenant_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    details: dict[str, Any] = Field(default_factory=dict)
