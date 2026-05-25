from __future__ import annotations

from os import environ

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    app_name: str = "ProjectMemory"
    environment: str = "development"

    api_token: str = "dev-token"
    tenant_header: str = "X-Tenant-ID"
    request_id_header: str = "X-Request-Id"

    llm_backend: str = "deterministic"
    episodic_store_backend: str = "memory"
    semantic_store_backend: str = "memory"
    queue_backend: str = "memory"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "user_episodic_stream"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "memory-consolidation"
    redis_processing_name: str = "memory-consolidation:processing"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-nano"
    openai_analysis_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    memory_limit: int = 3
    request_timeout_seconds: float = 30.0
    max_consolidation_attempts: int = 3

    @field_validator("environment", "llm_backend", "episodic_store_backend", "semantic_store_backend", "queue_backend")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.strip().lower()

    @classmethod
    def from_env(cls) -> "Settings":
        data = {
            "app_name": environ.get("PROJECTMEMORY_APP_NAME", cls.model_fields["app_name"].default),
            "environment": environ.get("PROJECTMEMORY_ENV", cls.model_fields["environment"].default),
            "api_token": environ.get("PROJECTMEMORY_API_TOKEN", cls.model_fields["api_token"].default),
            "tenant_header": environ.get("PROJECTMEMORY_TENANT_HEADER", cls.model_fields["tenant_header"].default),
            "request_id_header": environ.get(
                "PROJECTMEMORY_REQUEST_ID_HEADER", cls.model_fields["request_id_header"].default
            ),
            "llm_backend": environ.get("PROJECTMEMORY_LLM_BACKEND", cls.model_fields["llm_backend"].default),
            "episodic_store_backend": environ.get(
                "PROJECTMEMORY_EPISODIC_BACKEND", cls.model_fields["episodic_store_backend"].default
            ),
            "semantic_store_backend": environ.get(
                "PROJECTMEMORY_SEMANTIC_BACKEND", cls.model_fields["semantic_store_backend"].default
            ),
            "queue_backend": environ.get("PROJECTMEMORY_QUEUE_BACKEND", cls.model_fields["queue_backend"].default),
            "qdrant_host": environ.get("PROJECTMEMORY_QDRANT_HOST", cls.model_fields["qdrant_host"].default),
            "qdrant_port": int(environ.get("PROJECTMEMORY_QDRANT_PORT", cls.model_fields["qdrant_port"].default)),
            "qdrant_collection": environ.get(
                "PROJECTMEMORY_QDRANT_COLLECTION", cls.model_fields["qdrant_collection"].default
            ),
            "neo4j_uri": environ.get("PROJECTMEMORY_NEO4J_URI", cls.model_fields["neo4j_uri"].default),
            "neo4j_username": environ.get(
                "PROJECTMEMORY_NEO4J_USERNAME", cls.model_fields["neo4j_username"].default
            ),
            "neo4j_password": environ.get(
                "PROJECTMEMORY_NEO4J_PASSWORD", cls.model_fields["neo4j_password"].default
            ),
            "redis_url": environ.get("PROJECTMEMORY_REDIS_URL", cls.model_fields["redis_url"].default),
            "redis_queue_name": environ.get(
                "PROJECTMEMORY_REDIS_QUEUE_NAME", cls.model_fields["redis_queue_name"].default
            ),
            "redis_processing_name": environ.get(
                "PROJECTMEMORY_REDIS_PROCESSING_NAME", cls.model_fields["redis_processing_name"].default
            ),
            "openai_api_key": environ.get("OPENAI_API_KEY"),
            "openai_chat_model": environ.get(
                "PROJECTMEMORY_OPENAI_CHAT_MODEL", cls.model_fields["openai_chat_model"].default
            ),
            "openai_analysis_model": environ.get(
                "PROJECTMEMORY_OPENAI_ANALYSIS_MODEL", cls.model_fields["openai_analysis_model"].default
            ),
            "openai_embedding_model": environ.get(
                "PROJECTMEMORY_OPENAI_EMBEDDING_MODEL", cls.model_fields["openai_embedding_model"].default
            ),
            "memory_limit": int(environ.get("PROJECTMEMORY_MEMORY_LIMIT", cls.model_fields["memory_limit"].default)),
            "request_timeout_seconds": float(
                environ.get(
                    "PROJECTMEMORY_REQUEST_TIMEOUT_SECONDS",
                    cls.model_fields["request_timeout_seconds"].default,
                )
            ),
            "max_consolidation_attempts": int(
                environ.get(
                    "PROJECTMEMORY_MAX_CONSOLIDATION_ATTEMPTS",
                    cls.model_fields["max_consolidation_attempts"].default,
                )
            ),
        }
        settings = cls.model_validate(data)
        settings.validate_runtime()
        return settings

    def validate_runtime(self) -> None:
        if self.environment == "production" and self.api_token == "dev-token":
            raise ValueError("PROJECTMEMORY_API_TOKEN must be set in production.")
        if self.llm_backend == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when PROJECTMEMORY_LLM_BACKEND=openai.")
        if self.episodic_store_backend not in {"memory", "qdrant"}:
            raise ValueError("PROJECTMEMORY_EPISODIC_BACKEND must be 'memory' or 'qdrant'.")
        if self.semantic_store_backend not in {"memory", "neo4j"}:
            raise ValueError("PROJECTMEMORY_SEMANTIC_BACKEND must be 'memory' or 'neo4j'.")
        if self.queue_backend not in {"memory", "redis"}:
            raise ValueError("PROJECTMEMORY_QUEUE_BACKEND must be 'memory' or 'redis'.")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
