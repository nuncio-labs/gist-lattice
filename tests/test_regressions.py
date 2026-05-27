import unittest

from gistlattice.backends import GistLatticeContainer, InMemoryEpisodicStore, InMemoryQueueBroker, InMemorySemanticStore
from gistlattice.config import Settings
from gistlattice.models import MemoryAnalysis
from gistlattice.service import GistLatticeService


class CountingLLM:
    def __init__(self) -> None:
        self.analysis_calls = 0

    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        self.analysis_calls += 1
        return MemoryAnalysis(gist=f"analysis:{prompt}", valence=0.1, importance=0.5)


class RegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_honors_zero_limit(self) -> None:
        llm = CountingLLM()
        settings = Settings(environment="test", llm_factory=lambda _settings: llm)
        episodic = InMemoryEpisodicStore()
        semantic = InMemorySemanticStore()
        container = GistLatticeContainer(
            settings=settings,
            llm=llm,
            episodic_store=episodic,
            semantic_store=semantic,
            queue=InMemoryQueueBroker(),
        )
        service = GistLatticeService(container)

        await episodic.register_episode(
            tenant_id="tenant-a",
            user_id="user-a",
            interaction_id="seed-1",
            embedding=await llm.embed_text("seed memory"),
            text="seed",
            gist="seed memory",
            valence=0.2,
            importance=0.9,
        )

        result = await service.retrieve(
            tenant_id="tenant-a",
            user_id="user-a",
            query="seed memory",
            limit=0,
        )

        self.assertEqual(result.memory_hits, 0)
        self.assertEqual(result.documents, [])

    async def test_consolidate_returns_stored_analysis_on_repeat_calls(self) -> None:
        llm = CountingLLM()
        settings = Settings(environment="test", llm_factory=lambda _settings: llm)
        container = GistLatticeContainer(
            settings=settings,
            llm=llm,
            episodic_store=InMemoryEpisodicStore(),
            semantic_store=InMemorySemanticStore(),
            queue=InMemoryQueueBroker(),
        )
        service = GistLatticeService(container)

        job = await service.queue_consolidation(
            tenant_id="tenant-a",
            user_id="user-a",
            prompt="Keep this stable",
            response="Keep this stable",
            request_id="req-1",
        )

        first = await service.consolidate(job.job_id)
        second = await service.consolidate(job.job_id)

        self.assertEqual(first, second)
        self.assertEqual(llm.analysis_calls, 1)
