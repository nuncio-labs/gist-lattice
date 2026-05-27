# Getting Started

GistLattice is a Python library for agent memory management. The default runtime uses in-memory stores, but you must provide an LLM adapter through `Settings.llm_factory` or `Settings.llm_factory_path`.

## Install

```bash
pip install gistlattice
```

If you want external providers, install the matching extras:

```bash
pip install gistlattice[qdrant]
pip install gistlattice[neo4j]
pip install gistlattice[redis]
```

For LLM providers, install the matching extra and use one of the ready-made provider factories. See [Backends](./backends.md) for the adapter contract and [Provider Adapters](./providers.md) for the provider-specific helpers.

If you use the `qdrant` episodic backend, set `GISTLATTICE_QDRANT_VECTOR_SIZE` when you already know the embedding width for your adapter. Otherwise, GistLattice will create the collection from the first embedding it stores.

## Minimal Example

```python
import asyncio

from gistlattice import Settings, build_default_service


async def main() -> None:
    settings = Settings(
        environment="test",
        llm_factory_path="my_project.providers.build_my_provider_llm",
    )
    service = build_default_service(settings)

    retrieval = await service.retrieve(
        tenant_id="tenant-a",
        user_id="user-a",
        query="Help me plan my next task.",
    )
    print(retrieval.hydrated_context)

    job = await service.queue_consolidation(
        tenant_id="tenant-a",
        user_id="user-a",
        prompt="Help me plan my next task.",
        response="Here is a memory-bearing response from your own app.",
        request_id="req-123",
    )
    analysis = await service.consolidate(job.job_id)
    print(analysis.gist)


asyncio.run(main())
```

## What Happens

- `retrieve(...)` embeds the query and finds relevant episodic memories.
- `hydrate_context(...)` builds prompt-ready context from memory.
- `queue_consolidation(...)` stores a prompt/response pair for later processing.
- `consolidate(...)` analyzes the interaction and writes it into memory.

## When To Use Each Method

- Use `retrieve(...)` when your own app wants to decide how to format memory.
- Use `hydrate_context(...)` when you want a ready-made system prompt block.
- Use `queue_consolidation(...)` when your app already has a prompt and response pair to store.
- Use `consolidate(...)` when you want to process a queued job immediately.

## Next Step

Read [Architecture](./architecture.md) if you want to understand how the engine pieces fit together.
