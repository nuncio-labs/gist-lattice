"""OpenAI-backed usage walkthrough for GistLattice.

This example shows the intended production pattern:

1. install the OpenAI extra
2. set OPENAI_API_KEY
3. create Settings with a provider factory
4. use the service to retrieve, hydrate, and consolidate memory

The example intentionally keeps the memory data small so the flow is easy to follow.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistlattice import Settings, build_default_service
from gistlattice.providers import build_openai_llm


def build_provider_llm(settings: Settings):
    """Wrapper kept small on purpose so you can swap provider defaults in one place."""

    return build_openai_llm(settings)


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY before running this example. "
            "You may also set GISTLATTICE_OPENAI_CHAT_MODEL and "
            "GISTLATTICE_OPENAI_EMBEDDING_MODEL if you want custom model names."
        )

    settings = Settings(
        environment="production",
        llm_factory=build_provider_llm,
        memory_limit=4,
    )
    service = build_default_service(settings)

    tenant_id = "tenant-product"
    user_id = "user-007"

    retrieval = await service.retrieve(
        tenant_id=tenant_id,
        user_id=user_id,
        query="Summarize what the assistant should remember about the launch plan.",
    )
    print("\n=== Retrieval ===")
    print(retrieval.hydrated_context)
    print(f"memory_hits={retrieval.memory_hits}")

    hydrated_context = await service.hydrate_context(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt="Summarize what the assistant should remember about the launch plan.",
    )
    print("\n=== Hydrated context ===")
    print(hydrated_context)

    job = await service.queue_consolidation(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt="The launch plan needs a risk register and a single owner.",
        response="I will remember that the launch plan needs a risk register and a single owner.",
        request_id="req-openai-001",
    )
    print("\n=== Queued job ===")
    print(f"job_id={job.job_id}")

    analysis = await service.consolidate(job.job_id)
    print("\n=== Memory analysis ===")
    print(f"gist={analysis.gist}")
    print(f"valence={analysis.valence:.2f}")
    print(f"importance={analysis.importance:.2f}")
    print(f"structural_location={analysis.structural_location}")
    print(f"core_project={analysis.core_project}")

    final = await service.retrieve(
        tenant_id=tenant_id,
        user_id=user_id,
        query="What do I remember about the launch plan?",
    )
    print("\n=== Retrieval after consolidation ===")
    print(final.hydrated_context)
    print(f"memory_hits={final.memory_hits}")


if __name__ == "__main__":
    asyncio.run(main())
