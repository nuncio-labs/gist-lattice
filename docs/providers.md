# Provider Adapters

GistLattice ships ready-made adapter factories for the most common provider SDKs.

If you prefer configuration over wrapper functions, see `Settings.llm_provider` and `Settings.embedding_provider` in [Configuration](./configuration.md). The runtime can assemble those providers for you.

## Available Factories

| Provider | Factory | Required extra | Notes |
| --- | --- | --- | --- |
| OpenAI | `build_openai_llm(...)` | `pip install gistlattice[openai]` | Uses OpenAI for embeddings and memory analysis. |
| Gemini | `build_gemini_llm(...)` | `pip install gistlattice[gemini]` | Uses Gemini for embeddings and memory analysis. |
| Ollama | `build_ollama_llm(...)` | `pip install gistlattice[ollama]` | Uses Ollama for embeddings and memory analysis. |
| Anthropic | `build_anthropic_llm(...)` | `pip install gistlattice[anthropic]` plus an embedding provider | Uses Anthropic for memory analysis and a separate embedding adapter for embeddings. |

The embedding-only helpers are also available:

- `build_openai_embeddings(...)`
- `build_gemini_embeddings(...)`
- `build_ollama_embeddings(...)`

## OpenAI

```python
from gistlattice import Settings, build_default_service
from gistlattice.providers import build_openai_llm

service = build_default_service(
    Settings(
        llm_factory=build_openai_llm,
    )
)
```

Environment variables:

- `OPENAI_API_KEY`
- `GISTLATTICE_OPENAI_CHAT_MODEL`
- `GISTLATTICE_OPENAI_EMBEDDING_MODEL`

## Gemini

```python
from gistlattice import Settings, build_default_service
from gistlattice.providers import build_gemini_llm

service = build_default_service(
    Settings(
        llm_factory=build_gemini_llm,
    )
)
```

Environment variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `GISTLATTICE_GEMINI_MODEL`
- `GISTLATTICE_GEMINI_EMBEDDING_MODEL`

## Ollama

```python
from gistlattice import Settings, build_default_service
from gistlattice.providers import build_ollama_llm

service = build_default_service(
    Settings(
        llm_factory=build_ollama_llm,
    )
)
```

Environment variables:

- `GISTLATTICE_OLLAMA_HOST`
- `GISTLATTICE_OLLAMA_MODEL`
- `GISTLATTICE_OLLAMA_EMBEDDING_MODEL`

## Anthropic

Anthropic does not provide embeddings itself, so the adapter requires a separate embedding provider factory.

```python
from gistlattice import Settings, build_default_service
from gistlattice.providers import build_anthropic_llm, build_openai_embeddings


def build_my_anthropic_llm(settings):
    return build_anthropic_llm(
        settings,
        embedding_client=build_openai_embeddings(settings),
    )


service = build_default_service(
    Settings(
        llm_factory=build_my_anthropic_llm,
    )
)
```

For environment-driven setup, point `GISTLATTICE_ANTHROPIC_EMBEDDINGS_FACTORY_PATH` at an embedding factory such as:

- `gistlattice.providers.build_openai_embeddings`
- `gistlattice.providers.build_gemini_embeddings`
- `gistlattice.providers.build_ollama_embeddings`
- or your own factory

Environment variables:

- `ANTHROPIC_API_KEY`
- `GISTLATTICE_ANTHROPIC_MODEL`
- `GISTLATTICE_ANTHROPIC_EMBEDDINGS_FACTORY_PATH`

## Notes

- Each factory returns an object implementing the GistLattice LLM contract.
- You can still wrap these factories in your own function if you need custom defaults.
- For provider-driven selection, the default runtime can now assemble separate LLM and embedding providers from `Settings`.
- If you pair any provider with the `qdrant` episodic backend, keep the embedding dimension stable across runs or set `GISTLATTICE_QDRANT_VECTOR_SIZE` explicitly.
- For full runtime selection details, see [Backends](./backends.md).
