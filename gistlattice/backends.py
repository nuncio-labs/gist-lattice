from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings
from .models import ConsolidationJob, MemoryAnalysis
from .providers import build_configured_llm
from .storage import StorageProvider, get_storage_provider

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...
    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis | dict[str, Any]: ...


class QueueBroker(Protocol):
    async def ensure_ready(self) -> None: ...
    async def enqueue(self, job: ConsolidationJob) -> None: ...
    async def dequeue(self, timeout_seconds: int = 1) -> ConsolidationJob | None: ...
    async def ack(self, raw_job: str) -> None: ...
    async def nack(self, raw_job: str) -> None: ...
    async def recover(self) -> None: ...
    async def depth(self) -> int: ...
    async def close(self) -> None: ...


def _load_custom_llm(settings: Settings) -> LLMClient:
    factory = settings.llm_factory
    if factory is None:
        if not settings.llm_factory_path:
            if settings.llm_provider:
                return build_configured_llm(settings)
            raise ValueError(
                "GISTLATTICE_LLM_FACTORY_PATH, Settings.llm_factory, or Settings.llm_provider is required."
            )

        module_path, attr_name = settings.llm_factory_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        factory = getattr(module, attr_name)

    candidate = factory(settings) if callable(factory) else factory
    if isinstance(candidate, type):  # allow factories to return a class
        candidate = candidate(settings)
    for method_name in ("embed_text", "analyze_interaction"):
        if not hasattr(candidate, method_name):
            raise TypeError(f"Custom LLM factory must provide a client with `{method_name}`.")
    return candidate


class InMemoryQueueBroker:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._processing: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def enqueue(self, job: ConsolidationJob) -> None:
        await self._queue.put(job.model_dump_json())

    async def dequeue(self, timeout_seconds: int = 1) -> ConsolidationJob | None:
        try:
            raw = await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except (TimeoutError, asyncio.TimeoutError):
            return None
        self._processing.append(raw)
        return ConsolidationJob.model_validate_json(raw)

    async def ack(self, raw_job: str) -> None:
        if raw_job in self._processing:
            self._processing.remove(raw_job)

    async def nack(self, raw_job: str) -> None:
        if raw_job in self._processing:
            self._processing.remove(raw_job)
        await self._queue.put(raw_job)

    async def recover(self) -> None:
        while self._processing:
            await self._queue.put(self._processing.pop(0))

    async def depth(self) -> int:
        return self._queue.qsize() + len(self._processing)

    async def close(self) -> None:
        return None


class RedisQueueBroker:
    def __init__(self, settings: Settings) -> None:
        try:
            import redis.asyncio as redis
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError("redis package is not installed") from exc

        self._redis_module = redis
        self._settings = settings
        self._client = redis.from_url(settings.redis_url, decode_responses=True)

    async def ensure_ready(self) -> None:
        await self._client.ping()

    @property
    def redis_client(self) -> Any:
        return self._client

    async def enqueue(self, job: ConsolidationJob) -> None:
        await self._client.lpush(self._settings.redis_queue_name, job.model_dump_json())

    async def dequeue(self, timeout_seconds: int = 1) -> ConsolidationJob | None:
        raw = await self._client.brpoplpush(
            self._settings.redis_queue_name, self._settings.redis_processing_name, timeout=timeout_seconds
        )
        if raw is None:
            return None
        return ConsolidationJob.model_validate_json(raw)

    async def ack(self, raw_job: str) -> None:
        await self._client.lrem(self._settings.redis_processing_name, 1, raw_job)

    async def nack(self, raw_job: str) -> None:
        await self._client.lrem(self._settings.redis_processing_name, 1, raw_job)
        await self._client.lpush(self._settings.redis_queue_name, raw_job)

    async def recover(self) -> None:
        while True:
            raw = await self._client.rpop(self._settings.redis_processing_name)
            if raw is None:
                break
            await self._client.lpush(self._settings.redis_queue_name, raw)

    async def depth(self) -> int:
        queued = await self._client.llen(self._settings.redis_queue_name)
        processing = await self._client.llen(self._settings.redis_processing_name)
        return int(queued + processing)

    async def close(self) -> None:
        await self._client.aclose()


@dataclass(slots=True)
class GistLatticeContainer:
    settings: Settings
    llm: LLMClient
    storage: StorageProvider
    queue: QueueBroker
    job_store: dict[str, ConsolidationJob] = field(default_factory=dict)
    job_status: dict[str, str] = field(default_factory=dict)
    job_results: dict[str, MemoryAnalysis] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GistLatticeContainer":
        llm = _load_custom_llm(settings)
        storage = get_storage_provider(settings)

        if settings.queue_backend == "redis":
            queue: QueueBroker = RedisQueueBroker(settings)
        else:
            queue = InMemoryQueueBroker()

        return cls(
            settings=settings,
            llm=llm,
            storage=storage,
            queue=queue,
        )

    async def ensure_ready(self) -> dict[str, bool]:
        await self.storage.initialize()
        await self.queue.ensure_ready()
        return {"storage": True, "queue": True}

    async def close(self) -> None:
        await self.storage.close()
        await self.queue.close()
