import unittest
from datetime import datetime, timedelta, timezone

from memory_service.backends import DeterministicLLMClient, InMemoryEpisodicStore, InMemoryQueueBroker, InMemorySemanticStore
from memory_service.config import Settings
from memory_service.service import MemoryService
from memory_service.backends import ServiceContainer


class MemoryScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_hydration_includes_active_context_and_retained_gist(self) -> None:
        settings = Settings(environment="test")
        episodic = InMemoryEpisodicStore()
        semantic = InMemorySemanticStore()
        llm = DeterministicLLMClient()
        container = ServiceContainer(
            settings=settings,
            llm=llm,
            episodic_store=episodic,
            semantic_store=semantic,
            queue=InMemoryQueueBroker(),
        )
        service = MemoryService(container)

        await episodic.register_episode(
            tenant_id="tenant-a",
            user_id="user-a",
            interaction_id="seed-1",
            embedding=await llm.embed_text("Prepare the Paris launch plan"),
            text="seed",
            gist="Prepare the Paris launch plan",
            valence=0.2,
            importance=0.9,
        )
        await semantic.mutate_state_edge(
            tenant_id="tenant-a",
            user_id="user-a",
            relationship_type="CURRENT_STATE",
            new_value="Focused",
            entity_type="Psychological_Node",
        )

        hydrated, retained = await service.build_hydrated_prompt(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Please help with the Paris launch plan.",
        )

        self.assertIn("Previous Gist: Prepare the Paris launch plan", hydrated)
        self.assertIn("CURRENT_STATE: Focused", hydrated)
        self.assertEqual(len(retained), 1)

    async def test_old_low_importance_memories_are_filtered_out(self) -> None:
        store = InMemoryEpisodicStore()
        llm = DeterministicLLMClient()
        embedding = await llm.embed_text("stale memory")
        await store.register_episode(
            tenant_id="tenant-a",
            user_id="user-a",
            interaction_id="old-1",
            embedding=embedding,
            text="old",
            gist="Old memory",
            valence=-0.5,
            importance=0.1,
        )
        record = store._episodes[("tenant-a", "user-a")][0]
        record["last_accessed"] = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        record["embedding"] = [0.0] * len(embedding)
        record["importance"] = 0.0
        retained = await store.recall_relevant_gists(
            tenant_id="tenant-a",
            user_id="user-a",
            query_embedding=embedding,
            limit=3,
        )
        self.assertEqual(retained, [])
