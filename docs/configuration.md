# Configuration

GistLattice reads settings from environment variables through `Settings.from_env()`.

## Core Settings

| Variable | Purpose | Default |
| --- | --- | --- |
| `GISTLATTICE_APP_NAME` | Human-readable application name | `GistLattice` |
| `GISTLATTICE_ENV` | Runtime environment name | `development` |
| `GISTLATTICE_LLM_FACTORY_PATH` | Import path for a custom LLM factory | required for env-based setup |
| `GISTLATTICE_EPISODIC_BACKEND` | Episodic memory backend | `memory` |
| `GISTLATTICE_SEMANTIC_BACKEND` | Semantic memory backend | `memory` |
| `GISTLATTICE_QUEUE_BACKEND` | Consolidation queue backend | `memory` |
| `GISTLATTICE_MEMORY_LIMIT` | Default number of memories to retrieve | `3` |

## Python-Only Settings

| Field | Purpose |
| --- | --- |
| `Settings.llm_factory` | Direct Python callable for building a custom LLM client |

## Provider-Specific Settings

| Variable | Purpose | Requires extra |
| --- | --- | --- |
| `GISTLATTICE_QDRANT_HOST` | Qdrant host | `qdrant` |
| `GISTLATTICE_QDRANT_PORT` | Qdrant port | `qdrant` |
| `GISTLATTICE_QDRANT_COLLECTION` | Qdrant collection name | `qdrant` |
| `GISTLATTICE_NEO4J_URI` | Neo4j Bolt URI | `neo4j` |
| `GISTLATTICE_NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `GISTLATTICE_NEO4J_PASSWORD` | Neo4j password | `neo4j` |
| `GISTLATTICE_REDIS_URL` | Redis connection URL | `redis` |
| `GISTLATTICE_REDIS_QUEUE_NAME` | Redis queue key | `redis` |
| `GISTLATTICE_REDIS_PROCESSING_NAME` | Redis processing key | `redis` |

## Runtime Validation

The settings layer validates:

- supported backend names
- LLM factory presence through `GISTLATTICE_LLM_FACTORY_PATH` or `Settings.llm_factory`

## Recommended Combinations

- Local development: `memory` stores + your own lightweight adapter
- Durable episodic memory: `qdrant` + `neo4j` + `redis` + your own adapter
- Custom model provider: any provider wrapped with `Settings.llm_factory`

For ready-made provider helpers, see [Provider Adapters](./providers.md).

## Example

```bash
export GISTLATTICE_ENV=production
export GISTLATTICE_LLM_FACTORY_PATH=your_module.build_llm_client
export GISTLATTICE_EPISODIC_BACKEND=qdrant
export GISTLATTICE_SEMANTIC_BACKEND=neo4j
export GISTLATTICE_QUEUE_BACKEND=redis
```

If you are using GistLattice directly in Python, you can also pass a callable factory on `Settings.llm_factory` instead of using `GISTLATTICE_LLM_FACTORY_PATH`.
