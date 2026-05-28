from __future__ import annotations

import asyncio
import threading
from typing import Any

from .config import Settings
from .runtime import build_default_service
from .models import MemoryAnalysis
from .memory_buffer import MemoryBufferController, InMemoryBufferStore, RedisBufferStore
from .backends import RedisQueueBroker


class GistLattice:
    """
    A high-level client for easy interaction with the memory layer.
    """

    def __init__(self, provider: str = "openai", tenant_id: str = "default", user_id: str = "default", **kwargs: Any) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        settings = Settings(llm_provider=provider, **kwargs)
        self.service = build_default_service(settings)
        
        # Initialize the Buffer Store based on the configured queue backend
        if isinstance(self.service.container.queue, RedisQueueBroker):
            buffer_store = RedisBufferStore(self.service.container.queue.redis_client)
        else:
            buffer_store = InMemoryBufferStore()
            
        # Initialize the Memory Buffer Controller to buffer interactions
        self.memory_buffer = MemoryBufferController(
            worker_callback=self._handle_buffer_flush,
            store=buffer_store,
            use_embeddings=False  # Default to just max_messages limit as requested
        )
        
        # Lazy thread initialization for synchronous callers
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None

    def _ensure_sync_thread(self) -> None:
        """Lazily initialize a background thread to run the asyncio event loop for synchronous calls."""
        if self._sync_loop is None:
            self._sync_loop = asyncio.new_event_loop()
            self._sync_thread = threading.Thread(target=self._sync_loop.run_forever, daemon=True)
            self._sync_thread.start()

    async def _handle_buffer_flush(self, tenant_id: str, user_id: str, payload: list[dict[str, Any]]) -> None:
        """
        Callback for when the memory buffer hits its limits.
        Merges the buffered turns into a single conversational transcript.
        """
        if not payload:
            return
            
        import uuid
        transcript_lines = []
        for turn in payload:
            transcript_lines.append(f"User: {turn['prompt']}")
            transcript_lines.append(f"AI: {turn['response']}")
            transcript_lines.append("---")
            
        combined_transcript = "\n".join(transcript_lines)
        
        # We queue it in the background as a single chunk to eliminate per-turn mutations
        await self.service.queue_consolidation(
            tenant_id=tenant_id,
            user_id=user_id,
            prompt="[Buffered Conversational Transcript]",
            response=combined_transcript,
            request_id=uuid.uuid4().hex,
        )

    async def aremember(self, prompt: str, response: str, run_in_background: bool = False, bypass_buffer: bool = False) -> MemoryAnalysis | str | None:
        """
        Stores the interaction into memory asynchronously.
        By default, it buffers interactions to eliminate per-turn LLM database mutations.
        If bypass_buffer=True, it will immediately consolidate the single interaction.
        If run_in_background=True and bypass_buffer=True, enqueues the memory job and returns the job ID (str).
        """
        if not bypass_buffer:
            await self.memory_buffer.process_turn(self.tenant_id, self.user_id, prompt, response)
            return None

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

    def remember(self, prompt: str, response: str, run_in_background: bool = False, bypass_buffer: bool = False) -> MemoryAnalysis | str | None:
        """
        Synchronous wrapper for `aremember`. 
        Spins up a lightweight background thread to safely execute background buffer flushes.
        For best performance in high-throughput native async apps, use `aremember` instead.
        """
        self._ensure_sync_thread()
        future = asyncio.run_coroutine_threadsafe(
            self.aremember(prompt, response, run_in_background, bypass_buffer),
            self._sync_loop
        )
        return future.result()

    async def aretrieve(self, query: str, limit: int | None = None) -> Any:
        """
        Retrieves relevant episodic memory based on a query asynchronously.
        """
        return await self.service.retrieve(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            query=query,
            limit=limit,
        )

    def retrieve(self, query: str, limit: int | None = None) -> Any:
        """
        Synchronous wrapper for `aretrieve`.
        """
        self._ensure_sync_thread()
        future = asyncio.run_coroutine_threadsafe(
            self.aretrieve(query, limit),
            self._sync_loop
        )
        return future.result()

    async def ahydrate_context(self, prompt: str) -> str:
        """
        Hydrates the given prompt with memory context asynchronously.
        """
        return await self.service.hydrate_context(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            prompt=prompt,
        )

    def hydrate_context(self, prompt: str) -> str:
        """
        Synchronous wrapper for `ahydrate_context`.
        """
        self._ensure_sync_thread()
        future = asyncio.run_coroutine_threadsafe(
            self.ahydrate_context(prompt),
            self._sync_loop
        )
        return future.result()
