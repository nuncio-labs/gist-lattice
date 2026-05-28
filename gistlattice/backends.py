from __future__ import annotations

import asyncio
import importlib
import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .config import Settings
from .models import ConsolidationJob, MemoryAnalysis, MemoryDocument, MemoryGist, MemoryRetrievalResult
from .providers import build_configured_llm

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    left = left[:length]
    right = right[:length]
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class LLMClient(Protocol):
    async def embed_text(self, text: str) -> list[float]: ...

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis | dict[str, Any]: ...


class EpisodicStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def register_episode(
        self,
        *,
        tenant_id: str,
        user_id: str,
        interaction_id: str,
        embedding: list[float],
        text: str,
        gist: str,
        valence: float,
        importance: float,
    ) -> None: ...

    async def recall_relevant_gists(
        self, *, tenant_id: str, user_id: str, query_embedding: list[float], limit: int
    ) -> list[MemoryGist]: ...

    async def close(self) -> None: ...


class SemanticStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def get_active_user_context(self, *, tenant_id: str, user_id: str) -> dict[str, str]: ...

    async def mutate_state_edge(
        self, *, tenant_id: str, user_id: str, relationship_type: str, new_value: str, entity_type: str
    ) -> None: ...

    async def close(self) -> None: ...


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


@dataclass(slots=True)
class InMemoryEpisodicStore:
    collection_name: str
    _episodes: dict[tuple[str, str], list[dict[str, Any]]]
    _by_interaction: dict[tuple[str, str, str], dict[str, Any]]
    _lock: asyncio.Lock

    def __init__(self, collection_name: str = "user_episodic_stream") -> None:
        self.collection_name = collection_name
        self._episodes = defaultdict(list)
        self._by_interaction = {}
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        return None

    async def register_episode(
        self,
        *,
        tenant_id: str,
        user_id: str,
        interaction_id: str,
        embedding: list[float],
        text: str,
        gist: str,
        valence: float,
        importance: float,
    ) -> None:
        key = (tenant_id, user_id, interaction_id)
        async with self._lock:
            payload = self._by_interaction.get(key)
            record = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "interaction_id": interaction_id,
                "timestamp": _now().isoformat(),
                "last_accessed": _now().isoformat(),
                "raw_text": text,
                "gist": gist,
                "valence": valence,
                "importance": importance,
                "decay_rate": 0.4 if importance < 0.5 else 0.05,
                "embedding": list(embedding),
            }
            if payload is None:
                self._episodes[(tenant_id, user_id)].append(record)
                self._by_interaction[key] = record
            else:
                payload.update(record)

    async def recall_relevant_gists(
        self, *, tenant_id: str, user_id: str, query_embedding: list[float], limit: int
    ) -> list[MemoryGist]:
        now = _now()
        ranked: list[tuple[float, dict[str, Any]]] = []
        async with self._lock:
            for record in self._episodes.get((tenant_id, user_id), []):
                last_accessed = datetime.fromisoformat(record["last_accessed"])
                days_elapsed = (now - last_accessed).total_seconds() / 86400.0
                strength = record["importance"] * math.exp(-record["decay_rate"] * days_elapsed)
                similarity = cosine_similarity(record["embedding"], query_embedding)
                score = (0.7 * similarity) + (0.3 * strength)
                if score > 0.15:
                    ranked.append((score, record))
            ranked.sort(key=lambda item: item[0], reverse=True)
            results: list[MemoryGist] = []
            for score, record in ranked[:limit]:
                record["last_accessed"] = now.isoformat()
                record["importance"] = min(1.0, record["importance"] + 0.08)
                results.append(
                    MemoryGist(
                        gist=record["gist"],
                        valence=record["valence"],
                        importance=record["importance"],
                        score=score,
                        raw_text=record["raw_text"],
                        last_accessed=now,
                    )
                )
        return results

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class InMemorySemanticStore:
    _state: dict[tuple[str, str], dict[str, str]]
    _history: dict[tuple[str, str], list[dict[str, Any]]]
    _lock: asyncio.Lock

    def __init__(self) -> None:
        self._state = defaultdict(dict)
        self._history = defaultdict(list)
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        return None

    async def get_active_user_context(self, *, tenant_id: str, user_id: str) -> dict[str, str]:
        async with self._lock:
            return dict(self._state[(tenant_id, user_id)])

    async def mutate_state_edge(
        self, *, tenant_id: str, user_id: str, relationship_type: str, new_value: str, entity_type: str
    ) -> None:
        key = (tenant_id, user_id)
        async with self._lock:
            current = self._state[key].get(relationship_type)
            if current == new_value:
                return
            if current is not None:
                self._history[key].append(
                    {
                        "superseded_type": relationship_type,
                        "archived_at": _now().isoformat(),
                        "value": current,
                        "entity_type": entity_type,
                    }
                )
            self._state[key][relationship_type] = new_value

    async def close(self) -> None:
        return None


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


class QdrantEpisodicStore:
    def __init__(self, settings: Settings) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError("qdrant-client package is not installed") from exc

        self._client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._collection = settings.qdrant_collection
        self._vector_size = settings.qdrant_vector_size
        self._distance = Distance
        self._filter = Filter
        self._field_condition = FieldCondition
        self._match_value = MatchValue
        self._point_struct = PointStruct
        self._vector_params = VectorParams

    async def ensure_ready(self) -> None:
        if await self._client.collection_exists(self._collection):
            return None
        if self._vector_size is None:
            return None
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._vector_params(size=self._vector_size, distance=self._distance.COSINE),
        )

    async def _ensure_collection_for_embedding(self, embedding: list[float]) -> None:
        if await self._client.collection_exists(self._collection):
            return None
        vector_size = self._vector_size or len(embedding)
        if vector_size <= 0:
            raise ValueError("Qdrant embeddings must have at least one dimension.")
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._vector_params(size=vector_size, distance=self._distance.COSINE),
        )

    async def register_episode(
        self,
        *,
        tenant_id: str,
        user_id: str,
        interaction_id: str,
        embedding: list[float],
        text: str,
        gist: str,
        valence: float,
        importance: float,
    ) -> None:
        await self._ensure_collection_for_embedding(embedding)
        payload = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "interaction_id": interaction_id,
            "timestamp": _now().isoformat(),
            "last_accessed": _now().isoformat(),
            "raw_text": text,
            "gist": gist,
            "valence": valence,
            "importance": importance,
            "decay_rate": 0.4 if importance < 0.5 else 0.05,
        }
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{tenant_id}:{user_id}:{interaction_id}"))
        await self._client.upsert(
            collection_name=self._collection,
            points=[self._point_struct(id=point_id, vector=embedding, payload=payload)],
        )

    async def recall_relevant_gists(
        self, *, tenant_id: str, user_id: str, query_embedding: list[float], limit: int
    ) -> list[MemoryGist]:
        if not await self._client.collection_exists(self._collection):
            return []
        results_object = await self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            query_filter=self._filter(
                must=[
                    self._field_condition(key="tenant_id", match=self._match_value(value=tenant_id)),
                    self._field_condition(key="user_id", match=self._match_value(value=user_id)),
                ]
            ),
            limit=limit * 2,
        )
        now = _now()
        memories: list[MemoryGist] = []
        for result in results_object.points[:limit]:
            payload = result.payload
            last_accessed = datetime.fromisoformat(payload["last_accessed"])
            days_elapsed = (now - last_accessed).total_seconds() / 86400.0
            score = payload["importance"] * math.exp(-payload["decay_rate"] * days_elapsed)
            memories.append(
                MemoryGist(
                    gist=payload["gist"],
                    valence=payload["valence"],
                    importance=payload["importance"],
                    score=score,
                    raw_text=payload["raw_text"],
                    last_accessed=now,
                )
            )
        return memories

    async def close(self) -> None:
        await self._client.close()


class Neo4jSemanticStore:
    def __init__(self, settings: Settings) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError("neo4j package is not installed") from exc

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._bootstrap_schema()

    def _bootstrap_schema(self) -> None:
        query = "CREATE CONSTRAINT user_tenant_id_unique IF NOT EXISTS FOR (u:User) REQUIRE (u.tenant_id, u.id) IS UNIQUE"
        with self._driver.session() as session:
            session.run(query)

    async def ensure_ready(self) -> None:
        return None

    async def get_active_user_context(self, *, tenant_id: str, user_id: str) -> dict[str, str]:
        query = """
        MATCH (u:User {id: $user_id, tenant_id: $tenant_id})-[r:CURRENT_STATE|ACTIVE_FOCUS|LOCATED_AT]->(target)
        WHERE target.tenant_id = $tenant_id
        RETURN type(r) as relationship, target.name as state_value
        """

        def run_query() -> dict[str, str]:
            with self._driver.session() as session:
                result = session.run(query, user_id=user_id, tenant_id=tenant_id)
                context: dict[str, str] = {}
                for record in result:
                    context[record["relationship"].lower()] = record["state_value"]
                return context

        return await asyncio.to_thread(run_query)

    async def mutate_state_edge(
        self, *, tenant_id: str, user_id: str, relationship_type: str, new_value: str, entity_type: str
    ) -> None:
        allowed_relationships = {"CURRENT_STATE", "ACTIVE_FOCUS", "LOCATED_AT"}
        if relationship_type not in allowed_relationships:
            raise ValueError(f"Unsupported relationship_type: {relationship_type}")

        query = f"""
        MERGE (u:User {{id: $user_id, tenant_id: $tenant_id}})
        OPTIONAL MATCH (u)-[old_rel:{relationship_type}]->(old_target)
        WHERE old_target.name <> $new_value
        FOREACH (x IN CASE WHEN old_rel IS NOT NULL THEN [1] ELSE [] END |
            CREATE (u)-[:HISTORICAL_ARCHIVE {{
                superseded_type: '{relationship_type}',
                archived_at: timestamp(),
                value: old_target.name
            }}]->(old_target)
            DELETE old_rel
        )
        MERGE (new_target:Entity {{tenant_id: $tenant_id, name: $new_value, type: $entity_type}})
        MERGE (u)-[:{relationship_type}]->(new_target)
        """

        def run_query() -> None:
            with self._driver.session() as session:
                session.run(query, user_id=user_id, tenant_id=tenant_id, new_value=new_value, entity_type=entity_type)

        await asyncio.to_thread(run_query)

    async def close(self) -> None:
        await asyncio.to_thread(self._driver.close)


@dataclass(slots=True)
class GistLatticeContainer:
    settings: Settings
    llm: LLMClient
    episodic_store: EpisodicStore
    semantic_store: SemanticStore
    queue: QueueBroker
    job_store: dict[str, ConsolidationJob] = field(default_factory=dict)
    job_status: dict[str, str] = field(default_factory=dict)
    job_results: dict[str, MemoryAnalysis] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GistLatticeContainer":
        llm = _load_custom_llm(settings)

        if settings.episodic_store_backend == "qdrant":
            episodic_store: EpisodicStore = QdrantEpisodicStore(settings)
        else:
            episodic_store = InMemoryEpisodicStore(collection_name=settings.qdrant_collection)

        if settings.semantic_store_backend == "neo4j":
            semantic_store: SemanticStore = Neo4jSemanticStore(settings)
        else:
            semantic_store = InMemorySemanticStore()

        if settings.queue_backend == "redis":
            queue: QueueBroker = RedisQueueBroker(settings)
        else:
            queue = InMemoryQueueBroker()

        return cls(
            settings=settings,
            llm=llm,
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            queue=queue,
        )

    async def ensure_ready(self) -> dict[str, bool]:
        await self.episodic_store.ensure_ready()
        await self.semantic_store.ensure_ready()
        await self.queue.ensure_ready()
        return {"episodic_store": True, "semantic_store": True, "queue": True}

    async def close(self) -> None:
        await self.episodic_store.close()
        await self.semantic_store.close()
        await self.queue.close()
