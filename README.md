# GistLattice

GistLattice is a memory management library for agents and long-lived Python apps.

It helps you:
- retrieve relevant episodic memories
- hydrate prompts with durable semantic context
- consolidate interactions back into memory
- plug in different backends without changing your app logic

It also supports provider-agnostic custom LLM adapters, so you can wrap any model SDK or local runtime.
Ready-made adapters live in `gistlattice.providers` for OpenAI, Gemini, Ollama, and Anthropic.

The library is intentionally core-only for now. Use it directly from Python, and build your own API service later when you are ready to expose it to other languages.

## Install

Base install:

```bash
pip install gistlattice
```

Optional backend extras:

```bash
pip install gistlattice[qdrant]
pip install gistlattice[neo4j]
pip install gistlattice[redis]
```

## Quick Start

The default configuration uses in-memory backends, but you must still provide an LLM adapter through `Settings.llm_factory` or `Settings.llm_factory_path`.

```python
import asyncio

from gistlattice import Settings, build_default_service


async def main() -> None:
    service = build_default_service(
        Settings(
            environment="test",
            llm_factory_path="my_project.providers.build_my_provider_llm",
        )
    )

    retrieval = await service.retrieve(
        tenant_id="tenant-a",
        user_id="user-a",
        query="Help me plan my next task.",
    )
    print("Hydrated context:")
    print(retrieval.hydrated_context)

    job = await service.queue_consolidation(
        tenant_id="tenant-a",
        user_id="user-a",
        prompt="Help me plan my next task.",
        response="Here is a memory-bearing response from your own app.",
        request_id="req-123",
    )
    analysis = await service.consolidate(job.job_id)
    print("Analysis:")
    print(analysis.gist)


asyncio.run(main())
```

## Main Concepts

- `retrieve(...)` finds relevant memories for a tenant/user pair.
- `hydrate_context(...)` turns those memories into prompt-ready text.
- `queue_consolidation(...)` records a prompt/response pair for consolidation.
- `consolidate(...)` finalizes a queued memory job into episodic and semantic memory.

The service implementation lives in [gistlattice/service.py](/Users/sauravsinghal/Desktop/gist-lattice/gistlattice/service.py).

## Testing

Run the test suite with:

```bash
python3 -m unittest
```

Focused test files cover:
- runtime helpers
- configuration
- interaction flow
- memory scoring

## Project Layout

- `gistlattice/` core library code
- `tests/` unit tests
- `docs/` topic documentation
- `examples/` runnable walkthroughs

## Documentation Index

- [Docs Site](./docs/index.html)
- [Getting Started](./docs/getting-started.md)
- [Architecture](./docs/architecture.md)
- [Configuration](./docs/configuration.md)
- [Backends](./docs/backends.md)
- [Provider Adapters](./docs/providers.md)
- [Functions Reference](./docs/functions.md)
- [Examples](./examples/README.md)

The examples directory includes both a self-contained walkthrough and a provider-backed OpenAI example.
