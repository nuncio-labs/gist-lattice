from __future__ import annotations

from collections.abc import Callable
from os import environ
from typing import Any

from pydantic import BaseModel, field_validator

class Settings(BaseModel):
    app_name: str = "GistLattice"
    environment: str = "development"

    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None

    llm_factory_path: str | None = None
    llm_factory: Callable[["Settings"], Any] | None = None
    storage_backend: str = "memory"
    queue_backend: str = "memory"

    postgres_url: str = "postgresql://user:password@localhost:5432/gistlattice"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "memory-consolidation"
    redis_processing_name: str = "memory-consolidation:processing"

    memory_limit: int = 3

    @field_validator("environment", "storage_backend", "queue_backend")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.strip().lower()

    @classmethod
    def from_env(cls) -> "Settings":
        def env(name: str, default: str | int | float | None = None) -> str | int | float | None:
            value = environ.get(name)
            return default if value is None else value

        data = {
            "app_name": env("GISTLATTICE_APP_NAME", cls.model_fields["app_name"].default),
            "environment": env("GISTLATTICE_ENV", cls.model_fields["environment"].default),
            "llm_provider": env("GISTLATTICE_LLM_PROVIDER"),
            "llm_model": env("GISTLATTICE_LLM_MODEL"),
            "embedding_provider": env("GISTLATTICE_EMBEDDING_PROVIDER"),
            "embedding_model": env("GISTLATTICE_EMBEDDING_MODEL"),
            "llm_factory_path": env("GISTLATTICE_LLM_FACTORY_PATH"),
            "storage_backend": env("GISTLATTICE_STORAGE_BACKEND", cls.model_fields["storage_backend"].default),
            "queue_backend": env("GISTLATTICE_QUEUE_BACKEND", cls.model_fields["queue_backend"].default),
            "postgres_url": env("GISTLATTICE_POSTGRES_URL", cls.model_fields["postgres_url"].default),
            "neo4j_uri": env("GISTLATTICE_NEO4J_URI", cls.model_fields["neo4j_uri"].default),
            "neo4j_username": env("GISTLATTICE_NEO4J_USERNAME", cls.model_fields["neo4j_username"].default),
            "neo4j_password": env("GISTLATTICE_NEO4J_PASSWORD", cls.model_fields["neo4j_password"].default),
            "redis_url": env("GISTLATTICE_REDIS_URL", cls.model_fields["redis_url"].default),
            "redis_queue_name": env("GISTLATTICE_REDIS_QUEUE_NAME", cls.model_fields["redis_queue_name"].default),
            "redis_processing_name": env("GISTLATTICE_REDIS_PROCESSING_NAME", cls.model_fields["redis_processing_name"].default),
            "memory_limit": int(env("GISTLATTICE_MEMORY_LIMIT", cls.model_fields["memory_limit"].default)),
        }
        settings = cls.model_validate(data)
        return settings

    def model_post_init(self, __context: Any) -> None:
        self.validate_runtime()

    def validate_runtime(self) -> None:
        if not (self.llm_factory_path or self.llm_factory or self.llm_provider):
            raise ValueError("GISTLATTICE_LLM_FACTORY_PATH, Settings.llm_factory, or Settings.llm_provider is required.")
        if self.storage_backend not in {"memory", "postgres", "neo4j"}:
            raise ValueError("GISTLATTICE_STORAGE_BACKEND must be 'memory', 'postgres' or 'neo4j'.")
        if self.queue_backend not in {"memory", "redis"}:
            raise ValueError("GISTLATTICE_QUEUE_BACKEND must be 'memory' or 'redis'.")
        if self.embedding_provider == "anthropic":
            raise ValueError("Anthropic cannot be used as the embedding provider.")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
