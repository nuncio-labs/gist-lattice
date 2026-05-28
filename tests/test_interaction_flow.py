import unittest

from gistlattice.backends import InMemoryQueueBroker, GistLatticeContainer
from gistlattice.config import Settings
from gistlattice.service import GistLatticeService
from gistlattice.worker import ConsolidationWorker
from gistlattice.storage.memory import InMemoryStorageProvider
from tests.llm_factories import build_fake_provider_llm


class InteractionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_and_worker_consolidates(self) -> None:
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm, storage_backend="memory")
        container = GistLatticeContainer(
            settings=settings,
            llm=build_fake_provider_llm(settings),
            storage=InMemoryStorageProvider(),
            queue=InMemoryQueueBroker(),
        )
        service = GistLatticeService(container)
        worker = ConsolidationWorker(service)

        job = await service.queue_consolidation(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Urgent project update for Paris deployment",
            response="Context-aware reply: Urgent project update for Paris deployment",
            request_id="req-123",
        )

        self.assertEqual(job.user_id, "user-a")
        self.assertEqual(job.tenant_id, "tenant-a")

        processed = await worker.process_once()
        self.assertTrue(processed)
        
        chunks = container.storage._chunks.get(("tenant-a", "user-a"), [])
        self.assertEqual(len(chunks), 1)
        
        chunk = chunks[0]
        context = chunk.get("relationships", {})

        self.assertIn("CURRENT_STATE", context)
        self.assertIn("ACTIVE_FOCUS", context)
        self.assertIn("LOCATED_AT", context)

    async def test_worker_acknowledges_jobs_without_losing_them(self) -> None:
        settings = Settings(environment="test", llm_factory=build_fake_provider_llm, storage_backend="memory")
        container = GistLatticeContainer(
            settings=settings,
            llm=build_fake_provider_llm(settings),
            storage=InMemoryStorageProvider(),
            queue=InMemoryQueueBroker(),
        )
        service = GistLatticeService(container)
        worker = ConsolidationWorker(service)

        await service.queue_consolidation(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Need a task plan",
            response="Context-aware reply: Need a task plan",
            request_id="req-1",
        )

        processed = await worker.process_once()
        self.assertTrue(processed)
        empty = await container.queue.dequeue(timeout_seconds=0.01)
        self.assertIsNone(empty)
