from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class InteractionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class InteractionResponse(BaseModel):
    response: str
    interaction_id: str
    job_id: str
    request_id: str
    tenant_id: str
    user_id: str
    memory_hits: int


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


class Principal(BaseModel):
    subject: str
    tenant_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    details: dict[str, Any] = Field(default_factory=dict)
