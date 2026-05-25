# ProjectMemory

ProjectMemory is a memory-augmented FastAPI service that separates:

- episodic recall in a vector store
- semantic state in a graph store
- consolidation work in a durable queue and worker

The codebase now ships with a clean service boundary, request authentication, request IDs, and health/readiness endpoints.

## Running locally

The default configuration uses in-memory adapters so the service can run without external infrastructure.

```bash
uvicorn server:app --reload
```

## Environment

Key settings are read from environment variables:

- `PROJECTMEMORY_ENV`
- `PROJECTMEMORY_API_TOKEN`
- `PROJECTMEMORY_LLM_BACKEND`
- `PROJECTMEMORY_EPISODIC_BACKEND`
- `PROJECTMEMORY_SEMANTIC_BACKEND`
- `PROJECTMEMORY_QUEUE_BACKEND`
- `PROJECTMEMORY_REDIS_URL`
- `OPENAI_API_KEY`

Use `PROJECTMEMORY_LLM_BACKEND=openai`, `PROJECTMEMORY_EPISODIC_BACKEND=qdrant`, `PROJECTMEMORY_SEMANTIC_BACKEND=neo4j`, and `PROJECTMEMORY_QUEUE_BACKEND=redis` for production deployment.

## API

- `GET /healthz`
- `GET /readyz`
- `POST /v1/interactions`

The interaction endpoint expects:

- `Authorization: Bearer <token>`
- `X-Tenant-ID: <tenant>`

Body:

```json
{
  "user_id": "user-123",
  "prompt": "Help me plan my next task."
}
```

## Worker

Run the consolidation worker separately in production:

```bash
python -m memory_service.worker
```

It consumes queued interaction jobs, extracts structured memory, and writes to the episodic and semantic stores idempotently.
