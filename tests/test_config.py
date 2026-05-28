import unittest

from gistlattice.config import Settings
from tests.llm_factories import build_fake_provider_llm


class ConfigTests(unittest.TestCase):
    def test_settings_validate_and_default_to_memory_backends(self) -> None:
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm)
        self.assertEqual(settings.storage_backend, "memory")
        self.assertEqual(settings.queue_backend, "memory")

    def test_settings_require_llm_factory(self) -> None:
        with self.assertRaises(ValueError):
            Settings(environment="test")
