# Functions Reference

This page covers the public symbols you should rely on when using GistLattice as a library.

## Public Surface

- `Settings`
- `build_default_container(...)`
- `build_default_service(...)`
- `GistLatticeContainer`
- `GistLatticeService`

## `Settings`

The configuration model for GistLattice.

Use it to:

- select the runtime environment
- provide an LLM adapter plus episodic, semantic, and queue backends
- configure provider-specific values
- provide a direct Python LLM factory when you do not want to use an import path

Example:

```python
from gistlattice import Settings

settings = Settings(
    environment="test",
    llm_factory_path="my_project.providers.build_my_provider_llm",
)
```

## `build_default_container(settings=None)`

Builds a `GistLatticeContainer` using either:

- the provided `Settings`
- or settings loaded from the environment

Use this when you want the configured backends but do not need the service wrapper yet.

## `build_default_service(settings=None)`

Builds a `GistLatticeService` using the default container wiring.

This is the easiest entry point for most applications.

Example:

```python
from gistlattice import Settings, build_default_service

service = build_default_service(
    Settings(
        environment="test",
        llm_factory_path="my_project.providers.build_my_provider_llm",
    )
)
```

## `GistLatticeContainer`

Holds the selected runtime components:

- LLM client
- episodic store
- semantic store
- queue broker

### `GistLatticeContainer.from_settings(settings)`

Builds the full runtime container from a `Settings` object.

This is where backend selection happens.

## `GistLatticeService`

The core memory engine.

### `retrieve(...)`

Fetches relevant memory and semantic state for a tenant/user/query.

Returns a retrieval result with:

- `documents`
- `hydrated_context`
- `memory_hits`

### `hydrate_context(...)`

Turns retrieved memory into a prompt-ready context string.

### `build_hydrated_prompt(...)`

Returns both:

- a hydrated prompt string
- the underlying structured memory gists

### `queue_consolidation(...)`

Creates a consolidation job and pushes it to the queue.

Use this after you already have a response and want memory consolidation to happen later.

### `consolidate(...)`

Processes a queued job and writes the resulting memory analysis into the long-term stores.

## Provider Helpers

The `gistlattice.providers` module includes ready-made adapter factories for common SDKs:

- `build_openai_llm(...)`
- `build_gemini_llm(...)`
- `build_ollama_llm(...)`
- `build_anthropic_llm(...)`

It also includes embedding-only helpers for cases like Anthropic, where you want a separate embedding provider:

- `build_openai_embeddings(...)`
- `build_gemini_embeddings(...)`
- `build_ollama_embeddings(...)`
