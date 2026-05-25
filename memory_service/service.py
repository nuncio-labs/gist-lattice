from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from .backends import ServiceContainer
from .models import ConsolidationJob, InteractionRequest, InteractionResponse, MemoryAnalysis, MemoryGist

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryService:
    container: ServiceContainer

    async def build_hydrated_prompt(self, *, tenant_id: str, user_id: str, prompt: str) -> tuple[str, list[MemoryGist]]:
        query_embedding = await self.container.llm.embed_text(prompt)
        retained_gists = await self.container.episodic_store.recall_relevant_gists(
            tenant_id=tenant_id,
            user_id=user_id,
            query_embedding=query_embedding,
            limit=self.container.settings.memory_limit,
        )
        active_context = await self.container.semantic_store.get_active_user_context(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        gist_lines = [
            f"- Previous Gist: {gist.gist} [Emotional Valence: {gist.valence:.2f}, Importance: {gist.importance:.2f}]"
            for gist in retained_gists
        ]
        context_lines = [f"{key.upper()}: {value}" for key, value in sorted(active_context.items())]

        hydrated_prompt = "\n".join(
            [
                "You are the reasoning engine of an authenticated memory service.",
                "Use the long-term memory context to respond accurately and concisely.",
                "",
                "Active Context: " + (", ".join(context_lines) if context_lines else "none"),
                "Retained Memory Timeline Elements:",
                *(gist_lines or ["- No Relevant Memories Recalled"]),
            ]
        )
        return hydrated_prompt, retained_gists

    async def interact(
        self,
        *,
        tenant_id: str,
        user_id: str,
        prompt: str,
        request_id: str,
    ) -> InteractionResponse:
        hydrated_prompt, retained_gists = await self.build_hydrated_prompt(
            tenant_id=tenant_id,
            user_id=user_id,
            prompt=prompt,
        )
        final_output_text = await self.container.llm.generate_reply(
            system_prompt=hydrated_prompt,
            user_prompt=prompt,
        )

        interaction_id = uuid4().hex
        job_id = uuid4().hex
        job = ConsolidationJob(
            job_id=job_id,
            interaction_id=interaction_id,
            tenant_id=tenant_id,
            user_id=user_id,
            prompt=prompt,
            response=final_output_text,
            request_id=request_id,
        )
        await self.container.queue.enqueue(job)

        logger.info(
            "interaction_enqueued",
            extra={
                "event": "interaction_enqueued",
                "tenant_id": tenant_id,
                "user_id": user_id,
                "interaction_id": interaction_id,
                "job_id": job_id,
            },
        )

        return InteractionResponse(
            response=final_output_text,
            interaction_id=interaction_id,
            job_id=job_id,
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_hits=len(retained_gists),
        )

    async def consolidate(self, job: ConsolidationJob) -> MemoryAnalysis:
        analysis = await self.container.llm.analyze_interaction(prompt=job.prompt, response=job.response)
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
