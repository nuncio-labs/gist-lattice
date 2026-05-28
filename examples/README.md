# GistLattice Examples

This directory contains examples demonstrating how to use the simplified `GistLattice` API in different configurations.

## Running the Basic Examples

You will need the appropriate API keys set in your environment (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) to run these.

```bash
# 1. Basic OpenAI with synchronous execution
python 01_basic_openai_sync.py

# 2. Gemini LLM + OpenAI Embeddings with asynchronous background queues
python 02_hybrid_gemini_async.py
```

## Production Examples (Redis, Neo4j, Qdrant)

If you are ready to test GistLattice with actual production databases, navigate to the `production/` subdirectory.

```bash
cd production/

# Bring up the required databases via Docker
docker compose up -d

# Run the programmatic and environment-variable examples
python 01_programmatic.py
python 02_environment_variables.py
```

See [production/README.md](production/README.md) for more details.
