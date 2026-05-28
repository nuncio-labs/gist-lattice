import unittest
from unittest.mock import patch

from gistlattice.backends import GistLatticeContainer
from gistlattice.config import Settings
from gistlattice.models import MemoryAnalysis
from gistlattice.providers import build_configured_llm
from tests.llm_factories import build_fake_provider_llm


class LLMSelectionTests(unittest.TestCase):
    def test_custom_llm_factory_path_is_loaded(self) -> None:
        settings = Settings(
            environment="test",
            llm_factory_path="tests.llm_factories.build_fake_provider_llm",
        )
        container = GistLatticeContainer.from_settings(settings)
        self.assertEqual(container.llm.__class__.__name__, "FakeProviderLLM")

    def test_custom_llm_factory_callable_is_loaded(self) -> None:
        settings = Settings(
            environment="test",
            llm_factory=build_fake_provider_llm,
        )
        container = GistLatticeContainer.from_settings(settings)
        self.assertEqual(container.llm.__class__.__name__, "FakeProviderLLM")

class ProviderSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_and_embedding_provider_can_be_selected_independently(self) -> None:
        class FakeAnalysisLLM:
            async def embed_text(self, text: str) -> list[float]:
                return [1.0]

            async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
                return MemoryAnalysis(gist="analysis", valence=0.2, importance=0.4)

        class FakeEmbeddingClient:
            async def embed_text(self, text: str) -> list[float]:
                return [2.0]

        settings = Settings(
            environment="test",
            llm_provider="openai",
            llm_model="gpt-4.1-mini",
            embedding_provider="gemini",
            embedding_model="gemini-embedding-001",
        )

        with (
            patch("gistlattice.providers._build_provider_llm", return_value=FakeAnalysisLLM()) as build_llm,
            patch("gistlattice.providers._build_provider_embeddings", return_value=FakeEmbeddingClient()) as build_embedder,
        ):
            client = build_configured_llm(settings)

        self.assertEqual(await client.embed_text("hello"), [2.0])
        analysis = await client.analyze_interaction(prompt="hi", response="there")
        self.assertEqual(analysis.gist, "analysis")
        build_llm.assert_called_once_with(settings, provider="openai", model="gpt-4.1-mini")
        build_embedder.assert_called_once_with(settings, provider="gemini", model="gemini-embedding-001")
