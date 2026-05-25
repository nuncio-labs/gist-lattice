from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import MemoryAnalysis


def _load_factory(path: str) -> Any:
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _callable_factory(path: str) -> Any:
    factory = _load_factory(path)
    return factory if callable(factory) else factory


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Expected JSON object from provider, got: {text!r}")
    return json.loads(cleaned[start : end + 1])


def _analysis_prompt(prompt: str, response: str) -> str:
    return (
        "Return a single JSON object with keys "
        "gist, valence, importance, structural_location, core_project. "
        "Use null for unknown optional fields. "
        "valence must be between -1 and 1 and importance between 0 and 1.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Assistant response:\n{response}\n"
    )


async def _threaded(func: Any, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


@dataclass(slots=True)
class _OpenAIEmbeddingClient:
    client: Any
    model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.embeddings.create, model=self.model, input=text, encoding_format="float")
        return list(response.data[0].embedding)


@dataclass(slots=True)
class _OpenAIProviderLLM:
    client: Any
    chat_model: str
    embedding_model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.embeddings.create, model=self.embedding_model, input=text, encoding_format="float")
        return list(response.data[0].embedding)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        result = await _threaded(
            self.client.responses.create,
            model=self.chat_model,
            input=_analysis_prompt(prompt, response),
        )
        data = _json_object(getattr(result, "output_text", ""))
        return MemoryAnalysis.model_validate(data)


@dataclass(slots=True)
class _GeminiEmbeddingClient:
    client: Any
    model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.models.embed_content, model=self.model, contents=text)
        return list(response.embeddings[0].values)


@dataclass(slots=True)
class _GeminiProviderLLM:
    client: Any
    analysis_model: str
    embedding_model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.models.embed_content, model=self.embedding_model, contents=text)
        return list(response.embeddings[0].values)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        result = await _threaded(self.client.models.generate_content, model=self.analysis_model, contents=_analysis_prompt(prompt, response))
        data = _json_object(getattr(result, "text", "") or "")
        return MemoryAnalysis.model_validate(data)


@dataclass(slots=True)
class _OllamaEmbeddingClient:
    client: Any
    model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.embed, model=self.model, input=text)
        return list(response["embeddings"][0] if isinstance(response, dict) else response.embeddings[0])


@dataclass(slots=True)
class _OllamaProviderLLM:
    client: Any
    chat_model: str
    embedding_model: str

    async def embed_text(self, text: str) -> list[float]:
        response = await _threaded(self.client.embed, model=self.embedding_model, input=text)
        return list(response["embeddings"][0] if isinstance(response, dict) else response.embeddings[0])

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        result = await _threaded(
            self.client.chat,
            model=self.chat_model,
            messages=[{"role": "user", "content": _analysis_prompt(prompt, response)}],
        )
        content = result["message"]["content"] if isinstance(result, dict) else result.message.content
        data = _json_object(content)
        return MemoryAnalysis.model_validate(data)


@dataclass(slots=True)
class _AnthropicProviderLLM:
    client: Any
    model: str
    embedder: Any

    async def embed_text(self, text: str) -> list[float]:
        return await self.embedder.embed_text(text)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        result = await _threaded(
            self.client.messages.create,
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": _analysis_prompt(prompt, response)}],
        )
        content_blocks = getattr(result, "content", [])
        text = "".join(getattr(block, "text", "") for block in content_blocks)
        data = _json_object(text)
        return MemoryAnalysis.model_validate(data)


def build_openai_embeddings(_settings: Settings | None = None) -> _OpenAIEmbeddingClient:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `openai` extra to use OpenAI embeddings.") from exc

    client = OpenAI()
    model = os.getenv("GISTLATTICE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return _OpenAIEmbeddingClient(client=client, model=model)


def build_openai_llm(_settings: Settings | None = None) -> _OpenAIProviderLLM:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `openai` extra to use OpenAI adapters.") from exc

    client = OpenAI()
    chat_model = os.getenv("GISTLATTICE_OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model = os.getenv("GISTLATTICE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return _OpenAIProviderLLM(client=client, chat_model=chat_model, embedding_model=embedding_model)


def build_gemini_embeddings(_settings: Settings | None = None) -> _GeminiEmbeddingClient:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `gemini` extra to use Gemini embeddings.") from exc

    client = genai.Client()
    model = os.getenv("GISTLATTICE_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    return _GeminiEmbeddingClient(client=client, model=model)


def build_gemini_llm(_settings: Settings | None = None) -> _GeminiProviderLLM:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `gemini` extra to use Gemini adapters.") from exc

    client = genai.Client()
    analysis_model = os.getenv("GISTLATTICE_GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model = os.getenv("GISTLATTICE_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    return _GeminiProviderLLM(client=client, analysis_model=analysis_model, embedding_model=embedding_model)


def build_ollama_embeddings(_settings: Settings | None = None) -> _OllamaEmbeddingClient:
    try:
        from ollama import Client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `ollama` extra to use Ollama embeddings.") from exc

    host = os.getenv("GISTLATTICE_OLLAMA_HOST")
    client = Client(host=host) if host else Client()
    model = os.getenv("GISTLATTICE_OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    return _OllamaEmbeddingClient(client=client, model=model)


def build_ollama_llm(_settings: Settings | None = None) -> _OllamaProviderLLM:
    try:
        from ollama import Client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `ollama` extra to use Ollama adapters.") from exc

    host = os.getenv("GISTLATTICE_OLLAMA_HOST")
    client = Client(host=host) if host else Client()
    chat_model = os.getenv("GISTLATTICE_OLLAMA_MODEL", "gemma3")
    embedding_model = os.getenv("GISTLATTICE_OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    return _OllamaProviderLLM(client=client, chat_model=chat_model, embedding_model=embedding_model)


def build_anthropic_llm(
    _settings: Settings | None = None,
    *,
    embedding_client: Any | None = None,
    embedding_factory: Any | None = None,
    embedding_factory_path: str | None = None,
) -> _AnthropicProviderLLM:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `anthropic` extra to use Anthropic adapters.") from exc

    embedder = embedding_client
    if embedder is None:
        factory = embedding_factory
        if factory is None:
            embedder_path = embedding_factory_path or os.getenv("GISTLATTICE_ANTHROPIC_EMBEDDINGS_FACTORY_PATH")
            if not embedder_path:
                raise ValueError(
                    "Anthropic requires an embedding provider. Pass `embedding_client`/`embedding_factory` or set "
                    "`GISTLATTICE_ANTHROPIC_EMBEDDINGS_FACTORY_PATH`."
                )
            factory = _callable_factory(embedder_path)

        embedder = factory(_settings) if callable(factory) else factory
    if not hasattr(embedder, "embed_text"):
        raise TypeError("Anthropic embedding factory must return an object with `embed_text`.")

    client = anthropic.Anthropic()
    model = os.getenv("GISTLATTICE_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return _AnthropicProviderLLM(client=client, model=model, embedder=embedder)


__all__ = [
    "build_anthropic_llm",
    "build_gemini_embeddings",
    "build_gemini_llm",
    "build_ollama_embeddings",
    "build_ollama_llm",
    "build_openai_embeddings",
    "build_openai_llm",
]
