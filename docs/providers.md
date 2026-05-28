# Supported Providers

GistLattice is heavily provider-agnostic. It ships with built-in support for the four major LLM providers. 

Switching your entire memory infrastructure from OpenAI to Gemini is as simple as changing one string: `provider="gemini"`.

## 1. OpenAI

The default provider. Supports both text generation (for memory reflection) and vector embeddings.

**Requirements:**
- API Key: `OPENAI_API_KEY`
- Installs: `pip install gistlattice[openai]`

```python
memory = GistLattice(
    provider="openai",
    llm_model="gpt-4o",                            # Optional override
    embedding_model="text-embedding-3-small"       # Optional override
)
```

## 2. Google Gemini

Supports both text generation and vector embeddings. Highly recommended for cost-effective memory analysis.

**Requirements:**
- API Key: `GEMINI_API_KEY`
- Installs: `pip install gistlattice[gemini]`

```python
memory = GistLattice(
    provider="gemini",
    llm_model="gemini-1.5-pro",                   # Optional override
    embedding_model="text-embedding-004"          # Optional override
)
```

## 3. Anthropic (Claude)

Anthropic supports incredible text generation but **does not provide a public embedding API**. If you select Anthropic as your LLM provider, you **must** specify a separate embedding provider (like OpenAI or Gemini).

**Requirements:**
- API Key: `ANTHROPIC_API_KEY` (and the key for your embedding provider)
- Installs: `pip install gistlattice[anthropic,openai]`

```python
memory = GistLattice(
    provider="anthropic",
    llm_model="claude-3-5-sonnet-latest",
    embedding_provider="openai"                   # Required!
)
```

## 4. Ollama (Local Models)

Ollama allows you to run open-source models completely locally. It supports both text generation and embeddings.

**Requirements:**
- The Ollama daemon running locally (usually on port 11434).
- Models pulled locally (e.g. `ollama pull llama3`, `ollama pull nomic-embed-text`).
- Installs: `pip install gistlattice[ollama]`

```python
memory = GistLattice(
    provider="ollama",
    llm_model="llama3",
    embedding_model="nomic-embed-text"
)
```
