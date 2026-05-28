# Examples

This directory contains runnable walkthroughs that show how to use GistLattice in real code.

## Deep Usage Walkthrough

- [deep_usage.py](./deep_usage.py)

This example shows:

- how to build a custom LLM adapter
- how to create `Settings`
- how to build the default service
- how to configure separate providers and models for LLMs and embeddings
- how to retrieve memory
- how to hydrate context
- how to queue and consolidate an interaction
- how memory changes after consolidation

Run it with:

```bash
python3 examples/deep_usage.py
```

Use the helper factories in `gistlattice.providers` when you want a real model provider. The walkthrough uses a tiny local adapter so the example stays runnable without any external service.

## Provider-Backed Walkthrough

- [openai_usage.py](./openai_usage.py)

This example shows the same core flow using the OpenAI provider helper. Set `OPENAI_API_KEY` first, then run:

```bash
python3 examples/openai_usage.py
```

The pattern is the same for Gemini, Ollama, or Anthropic:

- pick the matching factory in `gistlattice.providers`
- pass it through `Settings.llm_factory`
- call the service methods as usual
