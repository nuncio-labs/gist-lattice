from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .config import Settings
from .models import ConsolidationJob, MemoryAnalysis, MemoryGist

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_embedding(text: str, dimensions: int = 1536) -> list[float]:
    values: list[float] = []
    for index in range(dimensions):
        digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:4], "big", signed=False)
        values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
    return values


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

    async def generate_reply(self, *, system_prompt: str, user_prompt: str) -> str: ...

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis: ...


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

    async def close(self) -> None: ...


class DeterministicLLMClient:
    async def embed_text(self, text: str) -> list[float]:
        return _stable_embedding(text)

    async def generate_reply(self, *, system_prompt: str, user_prompt: str) -> str:
        memory_line = ""
        for line in system_prompt.splitlines():
            if line.startswith("- Previous Gist:"):
                memory_line = line.split(":", 1)[1].strip()
                break
        focus_line = ""
        for line in system_prompt.splitlines():
            if line.startswith("Active Context:"):
                focus_line = line.split(":", 1)[1].strip()
                break
        pieces = [f"Context-aware reply: {user_prompt.strip()}"]
        if memory_line:
            pieces.append(f"memory={memory_line}")
        if focus_line and focus_line != "none":
            pieces.append(f"context={focus_line}")
        return " | ".join(pieces)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        text = f"{prompt} {response}".lower()
        importance = 0.35
        if any(word in text for word in ("urgent", "asap", "immediately", "important")):
            importance = 0.9
        elif "?" in prompt or "!" in prompt:
            importance = 0.65
        valence = 0.15
        if any(word in text for word in ("angry", "frustrated", "panic", "worried", "broken")):
            valence = -0.7
        elif any(word in text for word in ("happy", "great", "thanks", "relief", "solved")):
            valence = 0.6
        structural_location = None
        for candidate in ("new york", "san francisco", "london", "paris", "berlin", "tokyo", "delhi", "mumbai", "seattle"):
            if candidate in text:
                structural_location = candidate.title()
                break
        core_project = None
        for marker in ("project", "file", "task"):
            if marker in text:
                core_project = prompt.strip().splitlines()[0][:80]
                break
        gist = prompt.strip().splitlines()[0][:120]
        return MemoryAnalysis(
            gist=gist,
            valence=max(-1.0, min(1.0, valence)),
            importance=max(0.0, min(1.0, importance)),
            structural_location=structural_location,
            core_project=core_project,
        )


class OpenAILLMClient:
    def __init__(self, settings: Settings) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is installed
            raise RuntimeError("openai package is not installed") from exc

        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._settings = settings

    async def embed_text(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=[text],
            model=self._settings.openai_embedding_model,
        )
        return list(response.data[0].embedding)

    async def generate_reply(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._settings.openai_chat_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content or ""

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        analysis_prompt = (
            "Analyze the interaction and return JSON with keys gist, valence, importance, "
            "structural_location, and core_project. "
            f"Prompt: {prompt}\nResponse: {response}"
        )
        analysis = await self._client.chat.completions.create(
            model=self._settings.openai_analysis_model,
            messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(analysis.choices[0].message.content or "{}")
        return MemoryAnalysis.model_validate(payload)


@dataclass(slots=True)
class InMemoryEpisodicStore:
    collection_name: str
    _episodes: dict[tuple[str, str], list[dict[str, Any]]]
    _by_interaction: dict[tuple[str, str, str], dict[str, Any]]

    def __init__(self, collection_name: str = "user_episodic_stream") -> None:
        self.collection_name = collection_name
        self._episodes = defaultdict(list)
        self._by_interaction = {}

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

    def __init__(self) -> None:
        self._state = defaultdict(dict)
        self._history = defaultdict(list)

    async def ensure_ready(self) -> None:
        return None

    async def get_active_user_context(self, *, tenant_id: str, user_id: str) -> dict[str, str]:
        return dict(self._state[(tenant_id, user_id)])

    async def mutate_state_edge(
        self, *, tenant_id: str, user_id: str, relationship_type: str, new_value: str, entity_type: str
    ) -> None:
        key = (tenant_id, user_id)
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
        await self._client.rpush(self._settings.redis_queue_name, job.model_dump_json())

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
        await self._client.rpush(self._settings.redis_queue_name, raw_job)

    async def recover(self) -> None:
        while True:
            raw = await self._client.rpop(self._settings.redis_processing_name)
            if raw is None:
                break
            await self._client.rpush(self._settings.redis_queue_name, raw)

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
        self._distance = Distance
        self._filter = Filter
        self._field_condition = FieldCondition
        self._match_value = MatchValue
        self._point_struct = PointStruct
        self._vector_params = VectorParams

    async def ensure_ready(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._vector_params(size=1536, distance=self._distance.COSINE),
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
        await self.ensure_ready()
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
        point_id = int(hashlib.sha256(f"{tenant_id}:{user_id}:{interaction_id}".encode("utf-8")).hexdigest()[:16], 16)
        await self._client.upsert(
            collection_name=self._collection,
            points=[self._point_struct(id=point_id, vector=embedding, payload=payload)],
        )

    async def recall_relevant_gists(
        self, *, tenant_id: str, user_id: str, query_embedding: list[float], limit: int
    ) -> list[MemoryGist]:
        await self.ensure_ready()
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
        query = "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
        with self._driver.session() as session:
            session.run(query)

    async def ensure_ready(self) -> None:
        return None

    async def get_active_user_context(self, *, tenant_id: str, user_id: str) -> dict[str, str]:
        query = """
        MATCH (u:User {id: $user_id, tenant_id: $tenant_id})-[r:CURRENT_STATE|ACTIVE_FOCUS|LOCATED_AT]->(target)
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
        MERGE (new_target:Entity {{name: $new_value, type: $entity_type}})
        MERGE (u)-[:{relationship_type}]->(new_target)
        """

        def run_query() -> None:
            with self._driver.session() as session:
                session.run(query, user_id=user_id, tenant_id=tenant_id, new_value=new_value, entity_type=entity_type)

        await asyncio.to_thread(run_query)

    async def close(self) -> None:
        await asyncio.to_thread(self._driver.close)


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    llm: LLMClient
    episodic_store: EpisodicStore
    semantic_store: SemanticStore
    queue: QueueBroker

    @classmethod
    def from_settings(cls, settings: Settings) -> "ServiceContainer":
        if settings.llm_backend == "openai":
            llm: LLMClient = OpenAILLMClient(settings)
        else:
            llm = DeterministicLLMClient()

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
