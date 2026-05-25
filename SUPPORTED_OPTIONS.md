# Supported Options

This document lists the backends that work with GistLattice as of this repository state.
When you select a non-default backend, install the matching optional extra first.

## Core Runtime Modes

| Area | Option | Config | Notes |
| --- | --- | --- | --- |
| Library | In-process service | `gistlattice.service.GistLatticeService` | Primary runtime for embedding in Python apps |
| Worker | Consolidation worker | `python -m gistlattice.worker` | Processes queued consolidation jobs |
| Memory engine | In-memory | default | Works without external services for local development and tests |

## LLM Adapter

| Requirement | Config value | Requirements | Notes |
| --- | --- | --- | --- |
| Custom provider factory | required | `GISTLATTICE_LLM_FACTORY_PATH` or `Settings.llm_factory` | Any object with `embed_text` and `analyze_interaction` |
Custom provider factories can live anywhere importable by Python. The factory path must point to a callable or object that returns an LLM client implementation.

Ready-made provider helpers live in `gistlattice.providers`:

- `build_openai_llm(...)`
- `build_gemini_llm(...)`
- `build_ollama_llm(...)`
- `build_anthropic_llm(...)`

## Storage Backends

| Concern | Backend | Config value | Notes |
| --- | --- | --- | --- |
| Episodic memory | In-memory store | `memory` | Default local fallback |
| Episodic memory | Qdrant | `qdrant` + `qdrant` extra | Vector recall backend |
| Semantic memory | In-memory store | `memory` | Default local fallback |
| Semantic memory | Neo4j | `neo4j` + `neo4j` extra | Graph/state backend |
| Job queue | In-memory queue | `memory` | Default local fallback |
| Job queue | Redis queue | `redis` + `redis` extra | Durable queue for consolidation jobs |

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `GISTLATTICE_LLM_FACTORY_PATH` | Points to a custom LLM factory |
| `OPENAI_API_KEY` | OpenAI auth key |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini auth key |
| `ANTHROPIC_API_KEY` | Anthropic auth key |
| `GISTLATTICE_OLLAMA_HOST` | Ollama server host |
| `GISTLATTICE_ANTHROPIC_EMBEDDINGS_FACTORY_PATH` | Embedding factory for Anthropic |
| `GISTLATTICE_EPISODIC_BACKEND` | Selects episodic storage |
| `GISTLATTICE_SEMANTIC_BACKEND` | Selects semantic storage |
| `GISTLATTICE_QUEUE_BACKEND` | Selects queue backend |
| `GISTLATTICE_QDRANT_HOST` | Qdrant host |
| `GISTLATTICE_QDRANT_PORT` | Qdrant port |
| `GISTLATTICE_NEO4J_URI` | Neo4j Bolt URI |
| `GISTLATTICE_NEO4J_USERNAME` | Neo4j username |
| `GISTLATTICE_NEO4J_PASSWORD` | Neo4j password |
| `GISTLATTICE_REDIS_URL` | Redis connection URL |

## Practical Recommendations

- Use `Settings.llm_factory` in Python or `GISTLATTICE_LLM_FACTORY_PATH` in environment-driven deployments.
- Use `build_openai_llm(...)`, `build_gemini_llm(...)`, `build_ollama_llm(...)`, or `build_anthropic_llm(...)` when you want a ready-made provider adapter.
- Use `qdrant`, `neo4j`, and `redis` in production if you want durable episodic recall, semantic state, and queueing.
