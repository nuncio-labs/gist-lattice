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
class _CompositeProviderLLM:
    llm: Any
    embedder: Any

    async def embed_text(self, text: str) -> list[float]:
        return await self.embedder.embed_text(text)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        return await self.llm.analyze_interaction(prompt=prompt, response=response)


def _resolve_model(
    settings: Settings | None,
    *,
    provider: str,
    role: str,
    explicit_model: str | None = None,
) -> str:
    if explicit_model:
        return explicit_model
    if settings is not None:
        if role == "llm" and settings.llm_model:
            return settings.llm_model
        if role == "embedding" and settings.embedding_model:
            return settings.embedding_model

    if provider == "openai":
        return os.getenv(
            "GISTLATTICE_OPENAI_CHAT_MODEL" if role == "llm" else "GISTLATTICE_OPENAI_EMBEDDING_MODEL",
            "gpt-4.1-mini" if role == "llm" else "text-embedding-3-small",
        )
    if provider == "gemini":
        return os.getenv(
            "GISTLATTICE_GEMINI_MODEL" if role == "llm" else "GISTLATTICE_GEMINI_EMBEDDING_MODEL",
            "gemini-2.5-flash" if role == "llm" else "gemini-embedding-001",
        )
    if provider == "ollama":
        return os.getenv(
            "GISTLATTICE_OLLAMA_MODEL" if role == "llm" else "GISTLATTICE_OLLAMA_EMBEDDING_MODEL",
            "gemma3" if role == "llm" else "embeddinggemma",
        )
    if provider == "anthropic":
        if role == "embedding":
            raise ValueError("Anthropic does not provide embeddings.")
        return os.getenv("GISTLATTICE_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    raise ValueError(f"Unsupported provider: {provider}")


def _build_provider_llm(settings: Settings | None, *, provider: str, model: str) -> Any:
    if provider == "openai":
        return build_openai_llm(settings, chat_model=model, embedding_model=model)
    if provider == "gemini":
        return build_gemini_llm(settings, analysis_model=model, embedding_model=model)
    if provider == "ollama":
        return build_ollama_llm(settings, chat_model=model, embedding_model=model)
    if provider == "anthropic":
        return build_anthropic_llm(settings, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def _build_provider_embeddings(settings: Settings | None, *, provider: str, model: str) -> Any:
    if provider == "openai":
        return build_openai_embeddings(settings, model=model)
    if provider == "gemini":
        return build_gemini_embeddings(settings, model=model)
    if provider == "ollama":
        return build_ollama_embeddings(settings, model=model)
    if provider == "anthropic":
        raise ValueError("Anthropic does not provide embeddings.")
    raise ValueError(f"Unsupported provider: {provider}")


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


def build_openai_embeddings(
    settings: Settings | None = None,
    *,
    model: str | None = None,
) -> _OpenAIEmbeddingClient:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `openai` extra to use OpenAI embeddings.") from exc

    client = OpenAI()
    embedding_model = _resolve_model(settings, provider="openai", role="embedding", explicit_model=model)
    return _OpenAIEmbeddingClient(client=client, model=embedding_model)


def build_openai_llm(
    settings: Settings | None = None,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> _OpenAIProviderLLM:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `openai` extra to use OpenAI adapters.") from exc

    client = OpenAI()
    analysis_model = _resolve_model(settings, provider="openai", role="llm", explicit_model=chat_model)
    resolved_embedding_model = _resolve_model(
        settings,
        provider="openai",
        role="embedding",
        explicit_model=embedding_model,
    )
    return _OpenAIProviderLLM(client=client, chat_model=analysis_model, embedding_model=resolved_embedding_model)


def build_gemini_embeddings(
    settings: Settings | None = None,
    *,
    model: str | None = None,
) -> _GeminiEmbeddingClient:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `gemini` extra to use Gemini embeddings.") from exc

    client = genai.Client()
    embedding_model = _resolve_model(settings, provider="gemini", role="embedding", explicit_model=model)
    return _GeminiEmbeddingClient(client=client, model=embedding_model)


def build_gemini_llm(
    settings: Settings | None = None,
    *,
    analysis_model: str | None = None,
    embedding_model: str | None = None,
) -> _GeminiProviderLLM:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `gemini` extra to use Gemini adapters.") from exc

    client = genai.Client()
    resolved_analysis_model = _resolve_model(settings, provider="gemini", role="llm", explicit_model=analysis_model)
    resolved_embedding_model = _resolve_model(
        settings,
        provider="gemini",
        role="embedding",
        explicit_model=embedding_model,
    )
    return _GeminiProviderLLM(
        client=client,
        analysis_model=resolved_analysis_model,
        embedding_model=resolved_embedding_model,
    )


def build_ollama_embeddings(
    settings: Settings | None = None,
    *,
    model: str | None = None,
) -> _OllamaEmbeddingClient:
    try:
        from ollama import Client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `ollama` extra to use Ollama embeddings.") from exc

    host = os.getenv("GISTLATTICE_OLLAMA_HOST")
    client = Client(host=host) if host else Client()
    embedding_model = _resolve_model(settings, provider="ollama", role="embedding", explicit_model=model)
    return _OllamaEmbeddingClient(client=client, model=embedding_model)


def build_ollama_llm(
    settings: Settings | None = None,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> _OllamaProviderLLM:
    try:
        from ollama import Client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the `ollama` extra to use Ollama adapters.") from exc

    host = os.getenv("GISTLATTICE_OLLAMA_HOST")
    client = Client(host=host) if host else Client()
    analysis_model = _resolve_model(settings, provider="ollama", role="llm", explicit_model=chat_model)
    resolved_embedding_model = _resolve_model(
        settings,
        provider="ollama",
        role="embedding",
        explicit_model=embedding_model,
    )
    return _OllamaProviderLLM(client=client, chat_model=analysis_model, embedding_model=resolved_embedding_model)


def build_anthropic_llm(
    settings: Settings | None = None,
    *,
    model: str | None = None,
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

        embedder = factory(settings) if callable(factory) else factory
    if not hasattr(embedder, "embed_text"):
        raise TypeError("Anthropic embedding factory must return an object with `embed_text`.")

    client = anthropic.Anthropic()
    resolved_model = _resolve_model(settings, provider="anthropic", role="llm", explicit_model=model)
    return _AnthropicProviderLLM(client=client, model=resolved_model, embedder=embedder)


def build_configured_llm(settings: Settings) -> Any:
    if settings.llm_factory or settings.llm_factory_path:
        raise ValueError("build_configured_llm is for provider-based settings only.")
    if not settings.llm_provider:
        raise ValueError("Settings.llm_provider is required when no custom LLM factory is provided.")

    llm_provider = settings.llm_provider.strip().lower()
    embedding_provider = (settings.embedding_provider or llm_provider).strip().lower()

    analysis_model = _resolve_model(settings, provider=llm_provider, role="llm", explicit_model=settings.llm_model)
    embedding_model = _resolve_model(
        settings,
        provider=embedding_provider,
        role="embedding",
        explicit_model=settings.embedding_model,
    )

    if llm_provider == "anthropic":
        embedder = _build_provider_embeddings(settings, provider=embedding_provider, model=embedding_model)
        analysis_llm = build_anthropic_llm(settings, model=analysis_model, embedding_client=embedder)
        return analysis_llm

    analysis_llm = _build_provider_llm(settings, provider=llm_provider, model=analysis_model)
    if llm_provider == embedding_provider:
        if analysis_model == embedding_model:
            return analysis_llm
        return _CompositeProviderLLM(
            llm=analysis_llm,
            embedder=_build_provider_embeddings(settings, provider=embedding_provider, model=embedding_model),
        )

    if embedding_provider == "anthropic":
        raise ValueError("Anthropic cannot be used as the embedding provider.")

    embedder = _build_provider_embeddings(settings, provider=embedding_provider, model=embedding_model)
    return _CompositeProviderLLM(llm=analysis_llm, embedder=embedder)


__all__ = [
    "build_anthropic_llm",
    "build_configured_llm",
    "build_gemini_embeddings",
    "build_gemini_llm",
    "build_ollama_embeddings",
    "build_ollama_llm",
    "build_openai_embeddings",
    "build_openai_llm",
]
