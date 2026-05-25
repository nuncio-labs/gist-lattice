import unittest

from gistlattice import Settings, build_default_container, build_default_service
from tests.llm_factories import build_fake_provider_llm


class RuntimeHelperTests(unittest.TestCase):
    def test_default_builders_create_matching_service_and_container(self) -> None:
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm)
        container = build_default_container(settings)
        service = build_default_service(settings)

        self.assertEqual(container.settings.environment, "test")
        self.assertEqual(service.container.settings.environment, "test")
        self.assertEqual(service.container.settings.app_name, container.settings.app_name)
