from __future__ import annotations

from collections.abc import Callable
from os import environ
from typing import Any

from pydantic import BaseModel, field_validator

class Settings(BaseModel):
    app_name: str = "GistLattice"
    environment: str = "development"

    llm_factory_path: str | None = None
    llm_factory: Callable[["Settings"], Any] | None = None
    episodic_store_backend: str = "memory"
    semantic_store_backend: str = "memory"
    queue_backend: str = "memory"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "user_episodic_stream"
    qdrant_vector_size: int | None = None

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    redis_url: str = "redis://localhost:6379/0"
    redis_queue_name: str = "memory-consolidation"
    redis_processing_name: str = "memory-consolidation:processing"

    memory_limit: int = 3

    @field_validator("environment", "episodic_store_backend", "semantic_store_backend", "queue_backend")
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
            "llm_factory_path": env("GISTLATTICE_LLM_FACTORY_PATH"),
            "episodic_store_backend": env("GISTLATTICE_EPISODIC_BACKEND", cls.model_fields["episodic_store_backend"].default),
            "semantic_store_backend": env("GISTLATTICE_SEMANTIC_BACKEND", cls.model_fields["semantic_store_backend"].default),
            "queue_backend": env("GISTLATTICE_QUEUE_BACKEND", cls.model_fields["queue_backend"].default),
            "qdrant_host": env("GISTLATTICE_QDRANT_HOST", cls.model_fields["qdrant_host"].default),
            "qdrant_port": int(env("GISTLATTICE_QDRANT_PORT", cls.model_fields["qdrant_port"].default)),
            "qdrant_collection": env("GISTLATTICE_QDRANT_COLLECTION", cls.model_fields["qdrant_collection"].default),
            "qdrant_vector_size": (
                int(value)
                if (value := env("GISTLATTICE_QDRANT_VECTOR_SIZE")) is not None
                else cls.model_fields["qdrant_vector_size"].default
            ),
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
        if not (self.llm_factory_path or self.llm_factory):
            raise ValueError("GISTLATTICE_LLM_FACTORY_PATH or Settings.llm_factory is required.")
        if self.episodic_store_backend not in {"memory", "qdrant"}:
            raise ValueError("GISTLATTICE_EPISODIC_BACKEND must be 'memory' or 'qdrant'.")
        if self.semantic_store_backend not in {"memory", "neo4j"}:
            raise ValueError("GISTLATTICE_SEMANTIC_BACKEND must be 'memory' or 'neo4j'.")
        if self.queue_backend not in {"memory", "redis"}:
            raise ValueError("GISTLATTICE_QUEUE_BACKEND must be 'memory' or 'redis'.")
        if self.qdrant_vector_size is not None and self.qdrant_vector_size <= 0:
            raise ValueError("GISTLATTICE_QDRANT_VECTOR_SIZE must be a positive integer.")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
