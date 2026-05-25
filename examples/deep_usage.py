"""Deep usage walkthrough for GistLattice.

This example is intentionally verbose. It shows the full mental model:

1. provide an LLM adapter
2. build a Settings object
3. create the default service
4. retrieve memory before and after consolidating an interaction
5. hydrate a prompt from memory
6. queue and consolidate a job

Replace ``WalkthroughLLM`` with one of the provider factories in
``gistlattice.providers`` when you wire the library into a real app.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistlattice import Settings, build_default_service
from gistlattice.models import MemoryAnalysis


def _stable_embedding(text: str, dimensions: int = 12) -> list[float]:
    """Return a tiny, deterministic embedding for the walkthrough.

    This keeps the example runnable without any external SDK.
    The real app should swap this for a provider factory.
    """

    values: list[float] = []
    for index in range(dimensions):
        digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:4], "big", signed=False)
        values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
    return values


@dataclass(slots=True)
class WalkthroughLLM:
    """Minimal adapter that satisfies the GistLattice contract."""

    async def embed_text(self, text: str) -> list[float]:
        return _stable_embedding(text)

    async def analyze_interaction(self, *, prompt: str, response: str) -> MemoryAnalysis:
        text = f"{prompt} {response}".lower()
        valence = 0.15
        if any(word in text for word in ("angry", "frustrated", "panic", "worried", "broken")):
            valence = -0.6
        elif any(word in text for word in ("happy", "great", "thanks", "relief", "solved")):
            valence = 0.55

        structural_location = None
        for candidate in ("paris", "new york", "london", "tokyo", "delhi", "seattle"):
            if candidate in text:
                structural_location = candidate.title()
                break

        core_project = None
        for marker in ("project", "task", "file", "launch"):
            if marker in text:
                core_project = prompt.strip().splitlines()[0][:80]
                break

        return MemoryAnalysis(
            gist=f"Memory gist: {prompt[:60]}",
            valence=valence,
            importance=0.6 if "?" in prompt else 0.4,
            structural_location=structural_location,
            core_project=core_project,
        )


def build_walkthrough_llm(_settings: Settings) -> WalkthroughLLM:
    """Factory that returns the adapter the library will use."""

    return WalkthroughLLM()


def describe_retrieval(title: str, result) -> None:
    """Pretty-print the high-level parts of a retrieval result."""

    print(f"\n=== {title} ===")
    print(f"Query: {result.query}")
    print(f"Memory hits: {result.memory_hits}")
    print("Hydrated context:")
    print(result.hydrated_context)
    print("Documents:")
    for index, document in enumerate(result.documents, start=1):
        print(f"  {index}. {document.page_content}")
        print(f"     metadata={document.metadata}")


async def main() -> None:
    # The key rule: GistLattice needs an LLM adapter.
    settings = Settings(
        environment="test",
        llm_factory=build_walkthrough_llm,
        memory_limit=3,
    )
    service = build_default_service(settings)

    tenant_id = "tenant-a"
    user_id = "user-a"

    # 1) Start with an empty memory state.
    initial = await service.retrieve(
        tenant_id=tenant_id,
        user_id=user_id,
        query="What should I remember about the Paris launch?",
    )
    describe_retrieval("Initial retrieval", initial)

    # 2) Hydrate memory into a prompt block the agent can insert into its own prompt.
    hydrated_context = await service.hydrate_context(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt="What should I remember about the Paris launch?",
    )
    print("\n=== Hydrated context ===")
    print(hydrated_context)

    # 3) If you want the structured memory objects as well as the text, use build_hydrated_prompt.
    hydrated_prompt, retained_gists = await service.build_hydrated_prompt(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt="What should I remember about the Paris launch?",
    )
    print("\n=== Hydrated prompt ===")
    print(hydrated_prompt)
    print("Retained gists:")
    for gist in retained_gists:
        print(f"  - {gist.gist} (importance={gist.importance:.2f}, valence={gist.valence:.2f})")

    # 4) Queue a new interaction for later consolidation.
    job = await service.queue_consolidation(
        tenant_id=tenant_id,
        user_id=user_id,
        prompt="We need a rollout checklist for the Paris launch.",
        response="Absolutely. I'll keep the rollout checklist, target date, and owner in memory.",
        request_id="req-001",
    )
    print("\n=== Queued consolidation ===")
    print(f"job_id={job.job_id}")
    print(f"interaction_id={job.interaction_id}")

    # 5) Consolidate immediately so the example is self-contained.
    analysis = await service.consolidate(job.job_id)
    print("\n=== Consolidation analysis ===")
    print(f"gist={analysis.gist}")
    print(f"valence={analysis.valence:.2f}")
    print(f"importance={analysis.importance:.2f}")
    print(f"structural_location={analysis.structural_location}")
    print(f"core_project={analysis.core_project}")

    # 6) Retrieve again to see the retained memory and semantic context.
    final = await service.retrieve(
        tenant_id=tenant_id,
        user_id=user_id,
        query="What should I remember about the Paris launch?",
    )
    describe_retrieval("Retrieval after consolidation", final)

    # 7) In a real agent loop, your app would now insert hydrated_context into
    #    the system prompt and use the resulting memory hits to guide behavior.


if __name__ == "__main__":
    asyncio.run(main())
