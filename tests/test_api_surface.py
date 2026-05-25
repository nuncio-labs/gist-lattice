import unittest

from memory_service.api import create_app
from memory_service.config import Settings


class ApiSurfaceTests(unittest.TestCase):
    def test_routes_exist(self) -> None:
        app = create_app(Settings(environment="test"))
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        self.assertIn("/healthz", paths)
        self.assertIn("/readyz", paths)
        self.assertIn("/v1/interactions", paths)
