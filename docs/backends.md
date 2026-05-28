# Backends

GistLattice keeps the runtime modular so you can switch backend providers without rewriting your app.

## LLM Adapter

| Adapter requirement | When To Use | Install |
| --- | --- | --- |
| Custom provider adapter | Always required | no extra required |

You can either:

- provide a custom adapter via `Settings.llm_factory` or `Settings.llm_factory_path`
- set `Settings.llm_provider` and optional separate embedding provider/model fields

The library will build the configured adapter for you when provider-based settings are used.

For ready-made provider helpers, see [Provider Adapters](./providers.md).

## Custom Provider Adapters

Custom LLM adapters let you use any hosted API, local runtime, or hybrid stack without changing the core library.

Your factory must return an object with:

- `embed_text(text)`
- `analyze_interaction(prompt, response)`

You can wire it in one of two ways:

- set `Settings.llm_factory_path` for environment-driven setup
- pass a callable directly to `Settings.llm_factory` in Python code

If you want the library to assemble the provider objects for you, set:

- `Settings.llm_provider`
- `Settings.llm_model`
- `Settings.embedding_provider` if it differs from the analysis provider
- `Settings.embedding_model` if you want a non-default embedding model

Example:

```python
from gistlattice import Settings, build_default_service
from my_project.providers import build_my_provider_llm

settings = Settings(
    llm_factory=build_my_provider_llm,
)
service = build_default_service(settings)
```

## Episodic Memory Backends

| Backend | When To Use | Install |
| --- | --- | --- |
| `memory` | Local development and tests | no extra |
| `qdrant` | Durable vector recall | `pip install gistlattice[qdrant]` |

Qdrant-backed episodic memory now supports either of two sizing modes:

- set `GISTLATTICE_QDRANT_VECTOR_SIZE` when you already know the embedding width
- leave it unset to let GistLattice create the collection from the first stored embedding

If you use a custom LLM adapter, make sure its embedding dimensions stay stable for a given deployment.

## Semantic Memory Backends

| Backend | When To Use | Install |
| --- | --- | --- |
| `memory` | Local development and tests | no extra |
| `neo4j` | Durable semantic/state graph | `pip install gistlattice[neo4j]` |

## Queue Backends

| Backend | When To Use | Install |
| --- | --- | --- |
| `memory` | Local development and tests | no extra |
| `redis` | Durable consolidation queue | `pip install gistlattice[redis]` |

Both queue implementations preserve FIFO job ordering, so switching from memory to Redis should not change processing order.

## How Selection Works

The `GistLatticeContainer.from_settings(...)` classmethod reads the selected storage backend names from `Settings` and wires the matching implementations together.

For LLM providers, the library intentionally stays provider-agnostic. Your custom factory can wrap any SDK or local runtime you want, including hosted APIs, local models, or hybrid pipelines. The adapter is always required and is supplied separately through `Settings`.

See [Architecture](./architecture.md) for the full flow.
