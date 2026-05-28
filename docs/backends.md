# Production Backends

By default, GistLattice runs completely in-memory. This is fantastic for prototyping and unit tests, but data is lost the moment your Python process shuts down.

When you are ready to persist your AI's memories, you can connect GistLattice to powerful, scalable storage backends.

## 1. Install Dependencies

You must install the driver packages for the backends you intend to use:
```bash
pip install gistlattice[postgres,neo4j,redis]
```

## 2. Storage Providers

GistLattice unifies both vector-based episodic memories and graph-based semantic entity relationships into a single `StorageProvider` interface.

### A. PostgreSQL (pgvector)
Uses `asyncpg` to store memories in relational bridge tables and performs native HNSW vector similarity search.
- **Activate:** `storage_backend="postgres"`

### B. Neo4j
Maintains a living knowledge graph of concepts and relationships (e.g. `LOCATED_AT`), while utilizing Neo4j's native vector indices for memory retrieval.
- **Activate:** `storage_backend="neo4j"`

## 3. Queue Broker & Buffer (Redis)
Allows for asynchronous, non-blocking execution of memory reflection. Prevents your user-facing web requests from hanging while waiting for the LLM to analyze the conversation.

**Bonus Feature:** When Redis is activated, GistLattice automatically upgrades your short-term conversational Memory Buffer from local Python memory to a distributed Redis store. This allows you to scale multiple load-balanced web workers without losing context!

- **Activate:** `queue_backend="redis"`

---

## 4. Configuration Approaches

You can configure these backends in two ways: programmatically in Python, or globally via Environment Variables.

### Approach 1: Programmatic Configuration
Pass the configurations directly into the `GistLattice` client.

```python
from gistlattice import GistLattice

memory = GistLattice(
    provider="openai",
    
    # 1. Enable Backends
    storage_backend="postgres",
    queue_backend="redis",
    
    # 2. Provide Credentials
    postgres_url="postgresql://user:password@localhost:5432/gistlattice",
    redis_url="redis://localhost:6379/0"
)
```

### Approach 2: Environment Variables (Recommended)
Define everything in a `.env` file or your OS environment. The `GistLattice` client will automatically absorb them without needing manual configuration.

```bash
# .env file
GISTLATTICE_LLM_PROVIDER=openai

GISTLATTICE_STORAGE_BACKEND=postgres
GISTLATTICE_QUEUE_BACKEND=redis

GISTLATTICE_POSTGRES_URL=postgresql://user:password@localhost:5432/gistlattice
GISTLATTICE_REDIS_URL=redis://localhost:6379/0
```

```python
from gistlattice import GistLattice

# The client instantly picks up all settings from the environment!
memory = GistLattice()
```
