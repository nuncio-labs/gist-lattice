from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from .backends import GistLatticeContainer
from .models import ConsolidationJob, MemoryAnalysis, MemoryDocument, MemoryGist, MemoryRetrievalResult

logger = logging.getLogger(__name__)


def _hydration_prefix(hydrated_context: str) -> str:
    return "\n".join(
        [
            "You are the reasoning engine of an authenticated memory service.",
            "Use the long-term memory context to respond accurately and concisely.",
            "",
            hydrated_context,
        ]
    )


@dataclass(slots=True)
class GistLatticeService:
    container: GistLatticeContainer

    async def retrieve(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int | None = None,
    ) -> MemoryRetrievalResult:
        query_embedding = await self.container.llm.embed_text(query)
        memory_limit = limit or self.container.settings.memory_limit
        retained_gists = await self.container.episodic_store.recall_relevant_gists(
            tenant_id=tenant_id,
            user_id=user_id,
            query_embedding=query_embedding,
            limit=memory_limit,
        )
        active_context = await self.container.semantic_store.get_active_user_context(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        context_lines = [f"{key.upper()}: {value}" for key, value in sorted(active_context.items())]
        hydrated_context = "\n".join(
            [
                "Active Context: " + (", ".join(context_lines) if context_lines else "none"),
                "Retained Memory Timeline Elements:",
                *(
                    [
                        f"- Previous Gist: {gist.gist} [Emotional Valence: {gist.valence:.2f}, Importance: {gist.importance:.2f}]"
                        for gist in retained_gists
                    ]
                    or ["- No Relevant Memories Recalled"]
                ),
            ]
        )

        documents = [
            MemoryDocument(
                page_content=gist.raw_text or gist.gist,
                metadata={
                    "gist": gist.gist,
                    "valence": gist.valence,
                    "importance": gist.importance,
                    "score": gist.score,
                },
            )
            for gist in retained_gists
        ]
        return MemoryRetrievalResult(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            documents=documents,
            hydrated_context=hydrated_context,
            memory_hits=len(retained_gists),
        )

    async def hydrate_context(self, *, tenant_id: str, user_id: str, prompt: str) -> str:
        retrieval = await self.retrieve(tenant_id=tenant_id, user_id=user_id, query=prompt)
        return _hydration_prefix(retrieval.hydrated_context)

    async def build_hydrated_prompt(self, *, tenant_id: str, user_id: str, prompt: str) -> tuple[str, list[MemoryGist]]:
        retrieval = await self.retrieve(tenant_id=tenant_id, user_id=user_id, query=prompt)
        return (
            _hydration_prefix(retrieval.hydrated_context),
            [
                MemoryGist(
                    gist=document.metadata["gist"],
                    valence=document.metadata["valence"],
                    importance=document.metadata["importance"],
                    score=document.metadata["score"],
                    raw_text=document.page_content,
                )
                for document in retrieval.documents
            ],
        )

    async def queue_consolidation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        prompt: str,
        response: str,
        request_id: str,
        interaction_id: str | None = None,
        job_id: str | None = None,
    ) -> ConsolidationJob:
        interaction_id = interaction_id or uuid4().hex
        job_id = job_id or uuid4().hex
        job = ConsolidationJob(
            job_id=job_id,
            interaction_id=interaction_id,
            tenant_id=tenant_id,
            user_id=user_id,
            prompt=prompt,
            response=response,
            request_id=request_id,
        )
        self.container.job_store[job.job_id] = job
        self.container.job_status[job.job_id] = "queued"
        await self.container.queue.enqueue(job)
        return job

    async def consolidate(self, job_id: str) -> MemoryAnalysis:
        job = self.container.job_store.get(job_id)
        if job is None:
            raise KeyError(f"Unknown consolidation job: {job_id}")
        if self.container.job_status.get(job_id) == "completed":
            return await self.container.llm.analyze_interaction(prompt=job.prompt, response=job.response)

        analysis = await self.container.llm.analyze_interaction(prompt=job.prompt, response=job.response)
        await self.finalize_consolidation(job, analysis)
        self.container.job_status[job_id] = "completed"
        return analysis

    async def finalize_consolidation(self, job: ConsolidationJob, analysis: MemoryAnalysis) -> MemoryAnalysis:
        embedding = await self.container.llm.embed_text(f"{job.prompt}\n{job.response}")
        await self.container.episodic_store.register_episode(
            tenant_id=job.tenant_id,
            user_id=job.user_id,
            interaction_id=job.interaction_id,
            embedding=embedding,
            text=f"User: {job.prompt}\nAI: {job.response}",
            gist=analysis.gist,
            valence=analysis.valence,
            importance=analysis.importance,
        )

        if analysis.structural_location:
            await self.container.semantic_store.mutate_state_edge(
                tenant_id=job.tenant_id,
                user_id=job.user_id,
                relationship_type="LOCATED_AT",
                new_value=analysis.structural_location,
                entity_type="Geographical_Node",
            )
        if analysis.core_project:
            await self.container.semantic_store.mutate_state_edge(
                tenant_id=job.tenant_id,
                user_id=job.user_id,
                relationship_type="ACTIVE_FOCUS",
                new_value=analysis.core_project,
                entity_type="Task_Node",
            )

        mood_state = "Regulated/Balanced" if analysis.valence > -0.2 else "Elevated Panic State"
        await self.container.semantic_store.mutate_state_edge(
            tenant_id=job.tenant_id,
            user_id=job.user_id,
            relationship_type="CURRENT_STATE",
            new_value=mood_state,
            entity_type="Psychological_Node",
        )

        logger.info(
            "interaction_consolidated",
            extra={
                "event": "interaction_consolidated",
                "tenant_id": job.tenant_id,
                "user_id": job.user_id,
                "interaction_id": job.interaction_id,
                "job_id": job.job_id,
            },
        )
        return analysis
