import unittest
from datetime import datetime, timedelta, timezone

from gistlattice.backends import InMemoryQueueBroker, GistLatticeContainer
from gistlattice.storage.memory import InMemoryStorageProvider
from gistlattice.config import Settings
from gistlattice.models import ExtractedMemory
from gistlattice.service import GistLatticeService
from tests.llm_factories import build_fake_provider_llm


class MemoryScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_hydration_includes_active_context_and_retained_gist(self) -> None:
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm, storage_backend="memory")
        storage = InMemoryStorageProvider()
        llm = build_fake_provider_llm(settings)
        container = GistLatticeContainer(
            settings=settings,
            llm=llm,
            storage=storage,
            queue=InMemoryQueueBroker(),
        )
        service = GistLatticeService(container)

        embedding = await llm.embed_text("Prepare the Paris launch plan")
        mem = ExtractedMemory(
            tenant_id="tenant-a",
            user_id="user-a",
            interaction_id="seed-1",
            gist="Prepare the Paris launch plan",
            valence=0.2,
            importance=0.9,
            embedding=embedding,
            relationships={"CURRENT_STATE": "Focused"}
        )
        await storage.write_memory(mem)

        hydrated, retained = await service.build_hydrated_prompt(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Please help with the Paris launch plan.",
        )

        self.assertIn("Previous Gist: Prepare the Paris launch plan", hydrated)
        self.assertEqual(len(retained), 1)

    async def test_old_low_importance_memories_are_filtered_out(self) -> None:
        store = InMemoryStorageProvider()
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm, storage_backend="memory")
        llm = build_fake_provider_llm(settings)
        embedding = await llm.embed_text("stale memory")
        
        mem = ExtractedMemory(
            tenant_id="tenant-a",
            user_id="user-a",
            interaction_id="old-1",
            gist="Old memory",
            valence=-0.5,
            importance=0.1,
            embedding=embedding,
        )
        await store.write_memory(mem)
        
        # Mutate the internal timestamp for testing
        record = store._chunks[("tenant-a", "user-a")][0]
        record["last_accessed"] = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        record["embedding"] = [0.0] * len(embedding)
        record["importance"] = 0.0
        
        retained = await store.vector_search(
            tenant_id="tenant-a",
            user_id="user-a",
            query_vector=embedding,
            limit=3,
        )
        self.assertEqual(retained, [])
