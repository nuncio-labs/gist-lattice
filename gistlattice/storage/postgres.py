import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from gistlattice.models import ExtractedMemory, MemoryGist
from gistlattice.storage.base import StorageProvider

def _now() -> datetime:
    return datetime.now(timezone.utc)

class PostgresStorageProvider(StorageProvider):
    def __init__(self, postgres_url: str):
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError("asyncpg package is not installed. Please `pip install asyncpg`.") from exc
        self._url = postgres_url
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(self._url)
        async with self._pool.acquire() as conn:
            # Enable pgvector
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create Memory Chunks table
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id VARCHAR(255) PRIMARY KEY,
                tenant_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                interaction_id VARCHAR(255) NOT NULL,
                gist TEXT NOT NULL,
                valence FLOAT NOT NULL,
                importance FLOAT NOT NULL,
                embedding vector,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Create Entities table
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id SERIAL PRIMARY KEY,
                tenant_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                UNIQUE (tenant_id, name)
            );
            """)

            # Create Bridge table for Relationships
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id VARCHAR(255) REFERENCES memory_chunks(id) ON DELETE CASCADE,
                entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                relationship_type VARCHAR(255) NOT NULL,
                PRIMARY KEY (memory_id, entity_id, relationship_type)
            );
            """)
            
            # Create HNSW index on the embedding column
            # Note: We specify the vector dimension dynamically or assume it's created. pgvector allows omitting dimension in column.
            await conn.execute("""
            CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx ON memory_chunks USING hnsw (embedding vector_cosine_ops);
            """)

    async def write_memory(self, memory: ExtractedMemory) -> None:
        import uuid
        if not self._pool:
            raise RuntimeError("PostgresStorageProvider not initialized.")
            
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{memory.tenant_id}:{memory.user_id}:{memory.interaction_id}"))
        embedding_str = f"[{','.join(map(str, memory.embedding))}]"
        
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # 1. Insert Memory Chunk
                await conn.execute("""
                    INSERT INTO memory_chunks (id, tenant_id, user_id, interaction_id, gist, valence, importance, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                    ON CONFLICT (id) DO UPDATE SET 
                        gist = EXCLUDED.gist,
                        valence = EXCLUDED.valence,
                        importance = EXCLUDED.importance,
                        embedding = EXCLUDED.embedding
                """, memory_id, memory.tenant_id, memory.user_id, memory.interaction_id, memory.gist, memory.valence, memory.importance, embedding_str)
                
                # 2. Insert Entities and Relationships
                for relationship_type, entity_name in memory.relationships.items():
                    # Upsert Entity
                    entity_id = await conn.fetchval("""
                        INSERT INTO entities (tenant_id, name)
                        VALUES ($1, $2)
                        ON CONFLICT (tenant_id, name) DO UPDATE SET name = EXCLUDED.name
                        RETURNING id;
                    """, memory.tenant_id, entity_name)
                    
                    if not entity_id:
                        # Fallback if EXCLUDED doesn't return id on some pg versions
                        entity_id = await conn.fetchval("SELECT id FROM entities WHERE tenant_id = $1 AND name = $2", memory.tenant_id, entity_name)
                        
                    # Insert Bridge
                    await conn.execute("""
                        INSERT INTO memory_entities (memory_id, entity_id, relationship_type)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                    """, memory_id, entity_id, relationship_type)

    async def vector_search(self, user_id: str, tenant_id: str, query_vector: list[float], limit: int) -> list[MemoryGist]:
        if not self._pool:
            raise RuntimeError("PostgresStorageProvider not initialized.")
            
        embedding_str = f"[{','.join(map(str, query_vector))}]"
        
        async with self._pool.acquire() as conn:
            # We use <-> for L2 distance or <=> for cosine distance. 
            # pgvector's <=> operator represents cosine distance. We want highest similarity, so smallest distance.
            rows = await conn.fetch("""
                SELECT 
                    gist, valence, importance, last_accessed, 
                    1 - (embedding <=> $1::vector) AS cosine_similarity
                FROM memory_chunks
                WHERE tenant_id = $2 AND user_id = $3
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $4
            """, embedding_str, tenant_id, user_id, limit)
            
            results = []
            for row in rows:
                results.append(MemoryGist(
                    gist=row['gist'],
                    valence=row['valence'],
                    importance=row['importance'],
                    score=row['cosine_similarity'],
                    last_accessed=row['last_accessed']
                ))
            return results

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
