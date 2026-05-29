import asyncio
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from gistlattice.models import ExtractedMemory, MemoryGist
from gistlattice.storage.base import StorageProvider

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

class InMemoryStorageProvider(StorageProvider):
    def __init__(self):
        self._chunks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._entities: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._relationships: dict[str, dict[str, str]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def write_memory(self, memory: ExtractedMemory) -> None:
        async with self._lock:
            record = {
                "tenant_id": memory.tenant_id,
                "user_id": memory.user_id,
                "interaction_id": memory.interaction_id,
                "gist": memory.gist,
                "valence": memory.valence,
                "importance": memory.importance,
                "embedding": memory.embedding,
                "relationships": memory.relationships,
                "last_accessed": _now().isoformat(),
            }
            self._chunks[(memory.tenant_id, memory.user_id)].append(record)

    async def vector_search(self, user_id: str, tenant_id: str, query_vector: list[float], limit: int) -> list[MemoryGist]:
        now = _now()
        ranked: list[tuple[float, dict[str, Any]]] = []
        
        async with self._lock:
            for record in self._chunks.get((tenant_id, user_id), []):
                last_accessed = datetime.fromisoformat(record["last_accessed"])
                days_elapsed = (now - last_accessed).total_seconds() / 86400.0
                
                decay_rate = 0.4 if record["importance"] < 0.5 else 0.05
                strength = record["importance"] * math.exp(-decay_rate * days_elapsed)
                
                similarity = cosine_similarity(record["embedding"], query_vector)
                score = (0.7 * similarity) + (0.3 * strength)
                
                if score > 0.15:
                    ranked.append((score, record))
                    
            ranked.sort(key=lambda item: item[0], reverse=True)
            
            results: list[MemoryGist] = []
            for score, record in ranked[:limit]:
                record["last_accessed"] = now.isoformat()
                record["importance"] = min(1.0, record["importance"] + 0.08)
                
                results.append(MemoryGist(
                    gist=record["gist"],
                    valence=record["valence"],
                    importance=record["importance"],
                    score=score,
                    raw_text=record["gist"],  # fallback to gist
                    last_accessed=now,
                    relationships=record.get("relationships", {})
                ))
            return results

    async def close(self) -> None:
        pass
