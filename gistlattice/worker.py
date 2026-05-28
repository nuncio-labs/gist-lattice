from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .runtime import build_default_service
from .service import GistLatticeService

logger = logging.getLogger(__name__)


class ConsolidationWorker:
    def __init__(self, service: GistLatticeService) -> None:
        self.service = service

    async def process_once(self, timeout_seconds: int = 1) -> bool:
        job = await self.service.container.queue.dequeue(timeout_seconds=timeout_seconds)
        if job is None:
            return False
        raw_job = job.model_dump_json()
        try:
            await self.service.consolidate(job)
        except Exception:
            logger.exception(
                "consolidation_failed",
                extra={
                    "event": "consolidation_failed",
                    "tenant_id": job.tenant_id,
                    "user_id": job.user_id,
                    "interaction_id": job.interaction_id,
                    "job_id": job.job_id,
                },
            )
            await self.service.container.queue.nack(raw_job)
            return False
        else:
            await self.service.container.queue.ack(raw_job)
            return True

    async def run_forever(self, timeout_seconds: int = 1) -> None:
        await self.service.container.queue.recover()
        while True:
            processed = await self.process_once(timeout_seconds=timeout_seconds)
            if not processed:
                await asyncio.sleep(0.2)


async def run_worker(settings: Settings | None = None) -> None:
    runtime_settings = settings or Settings.from_env()
    service = build_default_service(runtime_settings)
    container = service.container
    await container.ensure_ready()
    worker = ConsolidationWorker(service)
    try:
        await worker.run_forever()
    finally:
        await container.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
