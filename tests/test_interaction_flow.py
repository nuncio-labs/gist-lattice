import unittest

from memory_service.backends import DeterministicLLMClient, InMemoryEpisodicStore, InMemoryQueueBroker, InMemorySemanticStore, ServiceContainer
from memory_service.config import Settings
from memory_service.models import ConsolidationJob
from memory_service.service import MemoryService
from memory_service.worker import ConsolidationWorker


class InteractionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_interaction_queues_and_worker_consolidates(self) -> None:
        settings = Settings(environment="test", api_token="secret-token")
        container = ServiceContainer(
            settings=settings,
            llm=DeterministicLLMClient(),
            episodic_store=InMemoryEpisodicStore(),
            semantic_store=InMemorySemanticStore(),
            queue=InMemoryQueueBroker(),
        )
        service = MemoryService(container)
        worker = ConsolidationWorker(service)

        response = await service.interact(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Urgent project update for Paris deployment",
            request_id="req-123",
        )

        self.assertEqual(response.user_id, "user-a")
        self.assertEqual(response.tenant_id, "tenant-a")
        self.assertGreaterEqual(response.memory_hits, 0)

        processed = await worker.process_once()
        self.assertTrue(processed)
        context = await container.semantic_store.get_active_user_context(
            tenant_id="tenant-a",
            user_id="user-a",
        )

        self.assertIn("CURRENT_STATE", context)
        self.assertIn("ACTIVE_FOCUS", context)
        self.assertIn("LOCATED_AT", context)

    async def test_worker_acknowledges_jobs_without_losing_them(self) -> None:
        settings = Settings(environment="test")
        container = ServiceContainer(
            settings=settings,
            llm=DeterministicLLMClient(),
            episodic_store=InMemoryEpisodicStore(),
            semantic_store=InMemorySemanticStore(),
            queue=InMemoryQueueBroker(),
        )
        service = MemoryService(container)
        worker = ConsolidationWorker(service)

        await container.queue.enqueue(
            ConsolidationJob(
                job_id="job-1",
                interaction_id="interaction-1",
                tenant_id="tenant-a",
                user_id="user-a",
                prompt="Need a task plan",
                response="Context-aware reply: Need a task plan",
                request_id="req-1",
            )
        )

        processed = await worker.process_once()
        self.assertTrue(processed)
        empty = await container.queue.dequeue(timeout_seconds=0.01)
        self.assertIsNone(empty)
