import unittest

from gistlattice.backends import GistLatticeContainer
from gistlattice.config import Settings
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
