from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from .backends import GistLatticeContainer
from .models import ConsolidationJob, MemoryAnalysis, MemoryDocument, MemoryGist, MemoryRetrievalResult, ExtractedMemory

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
        memory_limit = self.container.settings.memory_limit if limit is None else limit
        retained_gists = await self.container.storage.vector_search(
            tenant_id=tenant_id,
            user_id=user_id,
            query_vector=query_embedding,
            limit=memory_limit,
        )

        hydrated_context = "\n".join(
            [
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

    async def remember(
        self,
        *,
        tenant_id: str,
        user_id: str,
        prompt: str,
        response: str,
        interaction_id: str | None = None,
    ) -> MemoryAnalysis:
        job = ConsolidationJob(
            job_id=uuid4().hex,
            interaction_id=interaction_id or uuid4().hex,
            tenant_id=tenant_id,
            user_id=user_id,
            prompt=prompt,
            response=response,
            request_id=uuid4().hex,
        )
        return await self.consolidate(job)

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

    async def consolidate(self, job: ConsolidationJob | str) -> MemoryAnalysis:
        if isinstance(job, str):
            job_id = job
            job_obj = self.container.job_store.get(job_id)
            if job_obj is None:
                raise KeyError(f"Unknown consolidation job: {job_id}")
            job = job_obj
        else:
            job_id = job.job_id

        if self.container.job_status.get(job_id) == "completed":
            analysis = self.container.job_results.get(job_id)
            if analysis is not None:
                return analysis
            analysis = await self.container.llm.analyze_interaction(prompt=job.prompt, response=job.response)
            if isinstance(analysis, dict):
                analysis = MemoryAnalysis.model_validate(analysis)
            self.container.job_results[job_id] = analysis
            return analysis

        analysis = await self.container.llm.analyze_interaction(prompt=job.prompt, response=job.response)
        if isinstance(analysis, dict):
            analysis = MemoryAnalysis.model_validate(analysis)
        await self.finalize_consolidation(job, analysis)
        self.container.job_status[job_id] = "completed"
        return analysis

    async def finalize_consolidation(self, job: ConsolidationJob, analysis: MemoryAnalysis) -> MemoryAnalysis:
        embedding = await self.container.llm.embed_text(f"{job.prompt}\n{job.response}")
        
        relationships = {}
        if analysis.structural_location:
            relationships["LOCATED_AT"] = analysis.structural_location
        if analysis.core_project:
            relationships["ACTIVE_FOCUS"] = analysis.core_project
            
        mood_state = "Regulated/Balanced" if analysis.valence > -0.2 else "Elevated Panic State"
        relationships["CURRENT_STATE"] = mood_state
        
        extracted_memory = ExtractedMemory(
            tenant_id=job.tenant_id,
            user_id=job.user_id,
            interaction_id=job.interaction_id,
            gist=analysis.gist,
            valence=analysis.valence,
            importance=analysis.importance,
            embedding=embedding,
            relationships=relationships
        )
        
        await self.container.storage.write_memory(extracted_memory)

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
        self.container.job_results[job.job_id] = analysis
        return analysis
