from __future__ import annotations

from typing import Any

from .config import Settings
from .runtime import build_default_service
from .models import MemoryAnalysis


class GistLattice:
    """
    A high-level client for easy interaction with the memory layer.
    """

    def __init__(self, provider: str = "openai", tenant_id: str = "default", user_id: str = "default", **kwargs: Any) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        settings = Settings(llm_provider=provider, **kwargs)
        self.service = build_default_service(settings)

    async def remember(self, prompt: str, response: str, run_in_background: bool = False) -> MemoryAnalysis | str:
        """
        Stores the interaction into memory.
        If run_in_background=True, enqueues the memory job and returns the job ID (str).
        """
        if run_in_background:
            import uuid
            job = await self.service.queue_consolidation(
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                prompt=prompt,
                response=response,
                request_id=uuid.uuid4().hex,
            )
            return job.job_id

        return await self.service.remember(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            prompt=prompt,
            response=response,
        )

    async def retrieve(self, query: str, limit: int | None = None) -> Any:
        """
        Retrieves relevant episodic memory based on a query.
        """
        return await self.service.retrieve(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            query=query,
            limit=limit,
        )

    async def hydrate_context(self, prompt: str) -> str:
        """
        Hydrates the given prompt with memory context.
        """
        return await self.service.hydrate_context(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            prompt=prompt,
        )
