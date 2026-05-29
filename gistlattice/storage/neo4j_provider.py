import asyncio
from datetime import datetime, timezone
from typing import Any

from gistlattice.models import ExtractedMemory, MemoryGist
from gistlattice.storage.base import StorageProvider

def _now() -> datetime:
    return datetime.now(timezone.utc)

class Neo4jStorageProvider(StorageProvider):
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j package is not installed. Please `pip install neo4j`.") from exc
            
        self._driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password),
        )

    async def initialize(self) -> None:
        async with self._driver.session() as session:
            # Enforce unique user tenant id
            await session.run("CREATE CONSTRAINT user_tenant_id_unique IF NOT EXISTS FOR (u:User) REQUIRE (u.tenant_id, u.id) IS UNIQUE")
            # Create native vector index for memory chunks
            # In neo4j 5.15+, vector indices are created using db.index.vector.createNodeIndex
            # We will use the standard CREATE VECTOR INDEX syntax
            try:
                await session.run("""
                CREATE VECTOR INDEX memory_vector_index IF NOT EXISTS 
                FOR (m:MemoryChunk) ON (m.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 1536,
                  `vector.similarity_function`: 'cosine'
                }}
                """)
            except Exception as e:
                # Fallback if dimension is unknown or syntax fails.
                pass

    async def write_memory(self, memory: ExtractedMemory) -> None:
        import uuid
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{memory.tenant_id}:{memory.user_id}:{memory.interaction_id}"))
        
        # Single atomic Cypher query
        query = """
        MERGE (u:User {id: $user_id, tenant_id: $tenant_id})
        
        MERGE (m:MemoryChunk {id: $memory_id})
        SET m.tenant_id = $tenant_id,
            m.user_id = $user_id,
            m.interaction_id = $interaction_id,
            m.gist = $gist,
            m.valence = $valence,
            m.importance = $importance,
            m.embedding = $embedding,
            m.created_at = timestamp(),
            m.last_accessed = timestamp()
            
        MERGE (u)-[:HAS_MEMORY]->(m)
        """
        
        parameters = {
            "user_id": memory.user_id,
            "tenant_id": memory.tenant_id,
            "memory_id": memory_id,
            "interaction_id": memory.interaction_id,
            "gist": memory.gist,
            "valence": memory.valence,
            "importance": memory.importance,
            "embedding": memory.embedding
        }
        
        # Build relationship creation dynamically
        idx = 0
        for rel_type, entity_name in memory.relationships.items():
            safe_rel_type = rel_type.replace(" ", "_").upper()
            if not safe_rel_type.isalnum() and "_" not in safe_rel_type:
                safe_rel_type = "RELATED_TO"
                
            query += f"""
            MERGE (e{idx}:Entity {{tenant_id: $tenant_id, name: $entity_name_{idx}}})
            MERGE (m)-[:{safe_rel_type}]->(e{idx})
            """
            parameters[f"entity_name_{idx}"] = entity_name
            idx += 1

        async with self._driver.session() as session:
            await session.run(query, **parameters)

    async def vector_search(self, user_id: str, tenant_id: str, query_vector: list[float], limit: int) -> list[MemoryGist]:
        query = """
        CALL db.index.vector.queryNodes('memory_vector_index', $limit, $query_vector)
        YIELD node AS m, score
        WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
        OPTIONAL MATCH (m)-[r]->(e:Entity)
        RETURN m.gist AS gist, 
               m.valence AS valence, 
               m.importance AS importance, 
               score, 
               m.last_accessed AS last_accessed,
               collect([type(r), e.name]) AS relationships
        """
        results = []
        async with self._driver.session() as session:
            try:
                result = await session.run(query, limit=limit * 2, query_vector=query_vector, tenant_id=tenant_id, user_id=user_id)
                records = await result.data()
                
                for record in records[:limit]:
                    last_accessed_ts = record.get("last_accessed")
                    if last_accessed_ts:
                        last_accessed = datetime.fromtimestamp(last_accessed_ts / 1000.0, timezone.utc)
                    else:
                        last_accessed = _now()
                        
                    rels_raw = record.get("relationships", [])
                    relationships = {}
                    for rel in rels_raw:
                        if rel and len(rel) == 2 and rel[0] and rel[1]:
                            relationships[rel[0]] = rel[1]
                            
                    results.append(MemoryGist(
                        gist=record["gist"],
                        valence=record["valence"],
                        importance=record["importance"],
                        score=record["score"],
                        last_accessed=last_accessed,
                        relationships=relationships
                    ))
            except Exception as e:
                # If vector index fails (e.g. wrong dimension config), return empty.
                pass
                
        return results

    async def close(self) -> None:
        await self._driver.close()
