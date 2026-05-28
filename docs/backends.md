# Production Backends

By default, GistLattice runs completely in-memory. This is fantastic for prototyping and unit tests, but data is lost the moment your Python process shuts down.

When you are ready to persist your AI's memories, you can connect GistLattice to three powerful, scalable backends.

## 1. Install Dependencies

You must install the driver packages for the backends you intend to use:
```bash
pip install gistlattice[qdrant,neo4j,redis]
```

## 2. The Three Infrastructure Layers

GistLattice divides memory into three distinct architectural layers:

### A. Episodic Store (Qdrant)
Handles rapid vector-based retrieval of specific conversation chunks. 
- **Activate:** `episodic_store_backend="qdrant"`

### B. Semantic Store (Neo4j)
Maintains a living knowledge graph of concepts, states, and relationships (e.g., extracting that the user is located in "Paris" or currently focused on "Project X").
- **Activate:** `semantic_store_backend="neo4j"`

### C. Queue Broker (Redis)
Allows for asynchronous, non-blocking execution of memory reflection. Prevents your user-facing web requests from hanging while waiting for the LLM to analyze the conversation.
- **Activate:** `queue_backend="redis"`

---

## 3. Configuration Approaches

You can configure these backends in two ways: programmatically in Python, or globally via Environment Variables.

### Approach 1: Programmatic Configuration
Pass the configurations directly into the `GistLattice` client.

```python
from gistlattice import GistLattice

memory = GistLattice(
    provider="openai",
    
    # 1. Enable Backends
    episodic_store_backend="qdrant",
    semantic_store_backend="neo4j",
    queue_backend="redis",
    
    # 2. Provide Credentials
    qdrant_host="localhost",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="my_secure_password",
    redis_url="redis://localhost:6379/0"
)
```

### Approach 2: Environment Variables (Recommended)
Define everything in a `.env` file or your OS environment. The `GistLattice` client will automatically absorb them without needing manual configuration.

```bash
# .env file
GISTLATTICE_LLM_PROVIDER=openai

GISTLATTICE_EPISODIC_BACKEND=qdrant
GISTLATTICE_SEMANTIC_BACKEND=neo4j
GISTLATTICE_QUEUE_BACKEND=redis

GISTLATTICE_QDRANT_HOST=localhost
GISTLATTICE_NEO4J_URI=bolt://localhost:7687
GISTLATTICE_NEO4J_PASSWORD=my_secure_password
GISTLATTICE_REDIS_URL=redis://localhost:6379/0
```

```python
from gistlattice import GistLattice

# The client instantly picks up all settings from the environment!
memory = GistLattice()
```
