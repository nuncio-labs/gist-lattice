# GistLattice User Guide

Welcome to GistLattice! This guide serves as the master document for understanding how to use the library to grant your AI agents long-term, cognitive memory.

## The Concept of Cognitive Memory

Most vector databases simply dump raw chat transcripts into a database. GistLattice is different. It uses an LLM to actively **reflect** on a conversation before saving it.

When you ask GistLattice to remember something, it extracts three critical components:
1. **Gist**: A concise summary of the factual information.
2. **Valence**: The emotional tone of the interaction (-1.0 to 1.0). Did the user get angry? Were they excited? Your agent will remember their mood!
3. **Importance**: A score (0.0 to 1.0) dictating how crucial this memory is. Passing comments decay quickly; major life events are cemented permanently.

---

## 1. Initializing the Client

The entire surface area of the library is encapsulated in a single, elegant class: `GistLattice`.

```python
from gistlattice import GistLattice

# Initialize with OpenAI defaults
memory = GistLattice(
    provider="openai",
    tenant_id="my-organization",
    user_id="user-123"
)
```

> **Note:** `tenant_id` and `user_id` strictly isolate data. If you have 10,000 users, they can all share the same database cluster safely. GistLattice automatically filters queries so users never see each other's memories.

---

## 2. The 3 Core Methods

GistLattice exposes exactly three methods for interacting with memory.

### A. Saving a Memory: `remember()`

Takes a user prompt and an agent response, reflects on them, and saves the structured memory to the database.

**Synchronous execution (Default):**
```python
analysis = await memory.remember(
    prompt="I am really stressed about the product launch tomorrow.",
    response="I understand. Let's review the final checklist."
)
print(analysis.valence) # Output: -0.8 (Highly stressed)
```

**Asynchronous execution (High Traffic):**
If you have a high-traffic app, you don't want to wait for the LLM to analyze the memory. Set `run_in_background=True` and configure a Redis queue backend. It instantly returns a Job ID and processes the memory in the background!
```python
job_id = await memory.remember(
    prompt="...",
    response="...",
    run_in_background=True
)
```

### B. Injecting Context: `hydrate_context()`

When a user asks a new question, you need to pull relevant memories from the database and feed them into your LLM's system prompt.

```python
upcoming_query = "What should I focus on today?"

# Automatically searches for memories related to the upcoming query
# and formats them into a perfect System Prompt block.
system_prompt_addition = await memory.hydrate_context(upcoming_query)

# Inject this into your LLM call!
final_prompt = f"{system_prompt_addition}\n\nUser: {upcoming_query}"
```

### C. Raw Retrieval: `retrieve()`

If you want to build your own custom system prompt instead of using `hydrate_context()`, use `retrieve()` to fetch the raw data objects.

```python
results = await memory.retrieve(query="Product launch", limit=5)

for doc in results.documents:
    print(f"Gist: {doc.metadata['gist']}")
```

---

## 3. Next Steps

- Want to use Gemini, Anthropic, or Ollama instead of OpenAI? Check out the **[Providers Guide](providers.md)**.
- Ready to move off in-memory storage and connect to Redis, Neo4j, or Qdrant? Check out the **[Production Backends Guide](backends.md)**.
