import unittest

from fastapi import HTTPException

from memory_service.auth import build_principal_dependency
from memory_service.config import Settings


class ConfigAndAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_validate_and_default_to_memory_backends(self) -> None:
        settings = Settings(environment="test")
        self.assertEqual(settings.episodic_store_backend, "memory")
        self.assertEqual(settings.semantic_store_backend, "memory")
        self.assertEqual(settings.queue_backend, "memory")

    async def test_bearer_token_and_tenant_header_are_required(self) -> None:
        settings = Settings(environment="test", api_token="secret-token")
        require_principal = build_principal_dependency(settings)

        principal = await require_principal(
            authorization="Bearer secret-token",
            tenant_id="tenant-a",
        )
        self.assertEqual(principal.tenant_id, "tenant-a")

        with self.assertRaises(HTTPException):
            await require_principal(authorization="Bearer wrong", tenant_id="tenant-a")

        with self.assertRaises(HTTPException):
            await require_principal(authorization=None, tenant_id="tenant-a")
