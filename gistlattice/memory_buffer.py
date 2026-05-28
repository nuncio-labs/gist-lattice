import asyncio
import json
import logging
import math
from typing import List, Dict, Any, Callable, Awaitable, Protocol, TypedDict

logger = logging.getLogger(__name__)

class BufferState(TypedDict):
    active_buffer: List[Dict[str, Any]]
    buffer_embeddings: List[List[float]]
    potential_shift_buffer: List[Dict[str, Any]]
    potential_shift_embeddings: List[List[float]]
    hysteresis_counter: int

class BufferStore(Protocol):
    async def get_state(self, tenant_id: str, user_id: str) -> BufferState: ...
    async def save_state(self, tenant_id: str, user_id: str, state: BufferState) -> None: ...

class InMemoryBufferStore:
    def __init__(self):
        from collections import defaultdict
        self._states: dict[tuple[str, str], BufferState] = defaultdict(
            lambda: {
                "active_buffer": [],
                "buffer_embeddings": [],
                "potential_shift_buffer": [],
                "potential_shift_embeddings": [],
                "hysteresis_counter": 0,
            }
        )
        self._lock = asyncio.Lock()

    async def get_state(self, tenant_id: str, user_id: str) -> BufferState:
        async with self._lock:
            # Return a deep copy to avoid race conditions if mutated directly by caller
            state = self._states[(tenant_id, user_id)]
            return {
                "active_buffer": list(state["active_buffer"]),
                "buffer_embeddings": list(state["buffer_embeddings"]),
                "potential_shift_buffer": list(state["potential_shift_buffer"]),
                "potential_shift_embeddings": list(state["potential_shift_embeddings"]),
                "hysteresis_counter": state["hysteresis_counter"],
            }

    async def save_state(self, tenant_id: str, user_id: str, state: BufferState) -> None:
        async with self._lock:
            self._states[(tenant_id, user_id)] = {
                "active_buffer": list(state["active_buffer"]),
                "buffer_embeddings": list(state["buffer_embeddings"]),
                "potential_shift_buffer": list(state["potential_shift_buffer"]),
                "potential_shift_embeddings": list(state["potential_shift_embeddings"]),
                "hysteresis_counter": state["hysteresis_counter"],
            }

class RedisBufferStore:
    def __init__(self, redis_client: Any, ttl_seconds: int = 86400):
        self._client = redis_client
        self._ttl = ttl_seconds

    def _key(self, tenant_id: str, user_id: str) -> str:
        return f"gistlattice:buffer:{tenant_id}:{user_id}"

    async def get_state(self, tenant_id: str, user_id: str) -> BufferState:
        raw = await self._client.get(self._key(tenant_id, user_id))
        if not raw:
            return {
                "active_buffer": [],
                "buffer_embeddings": [],
                "potential_shift_buffer": [],
                "potential_shift_embeddings": [],
                "hysteresis_counter": 0,
            }
        return json.loads(raw)

    async def save_state(self, tenant_id: str, user_id: str, state: BufferState) -> None:
        raw = json.dumps(state)
        await self._client.setex(self._key(tenant_id, user_id), self._ttl, raw)


class MemoryBufferController:
    """
    Intelligent Dynamic Conversation Memory Buffer with Semantic Hysteresis and Context Overlap.
    """
    
    def __init__(
        self,
        worker_callback: Callable[[str, str, List[Dict[str, Any]]], Awaitable[None]],
        store: BufferStore,
        max_messages: int = 15,
        similarity_threshold: float = 0.65,
        hysteresis_threshold: int = 2,
        overlap_window_size: int = 2,
        use_embeddings: bool = False
    ):
        self.worker_callback = worker_callback
        self.store = store
        self.max_messages = max_messages
        self.similarity_threshold = similarity_threshold
        self.hysteresis_threshold = hysteresis_threshold
        self.overlap_window_size = overlap_window_size
        self.use_embeddings = use_embeddings

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _compute_centroid(embeddings: List[List[float]]) -> List[float]:
        if not embeddings:
            return []
        
        num_embeddings = len(embeddings)
        return [sum(dimension) / num_embeddings for dimension in zip(*embeddings)]

    async def _flush_to_worker(self, tenant_id: str, user_id: str, state: BufferState) -> None:
        if not state["active_buffer"]:
            return
            
        logger.info(f"Triggering flush to worker with {len(state['active_buffer'])} items.")
        
        # Create a safe copy of the payload to send to the worker
        payload = [dict(turn) for turn in state["active_buffer"]]
        
        # Asynchronously dispatch to worker
        asyncio.create_task(self._dispatch_to_worker(tenant_id, user_id, payload))
        
        # Apply overlapping context retention (The Cognitive Bridge)
        overlap_size = min(self.overlap_window_size, len(state["active_buffer"]))
        if overlap_size > 0:
            state["active_buffer"] = state["active_buffer"][-overlap_size:]
            state["buffer_embeddings"] = state["buffer_embeddings"][-overlap_size:]
            logger.debug(f"Retained {overlap_size} items for context overlap.")
        else:
            state["active_buffer"] = []
            state["buffer_embeddings"] = []

    async def _dispatch_to_worker(self, tenant_id: str, user_id: str, payload: List[Dict[str, Any]]) -> None:
        try:
            await self.worker_callback(tenant_id, user_id, payload)
        except Exception as e:
            logger.error(f"Error during background worker flush: {e}", exc_info=True)

    async def process_turn(
        self, 
        tenant_id: str,
        user_id: str,
        user_message: str, 
        agent_response: str, 
        msg_embedding: List[float] | None = None
    ) -> None:
        """
        Process a new conversation turn through the memory buffer logic.
        """
        turn_data = {"prompt": user_message, "response": agent_response}
        
        if self.use_embeddings and msg_embedding is None:
            raise ValueError("msg_embedding must be provided if use_embeddings is True")
            
        embedding_vec = list(msg_embedding) if msg_embedding else []

        state = await self.store.get_state(tenant_id, user_id)

        # A. HARD LIMIT CHECK
        if len(state["active_buffer"]) >= self.max_messages:
            logger.info("Hard limit reached. Flushing buffer.")
            await self._flush_to_worker(tenant_id, user_id, state)
            
            state["potential_shift_buffer"].clear()
            state["potential_shift_embeddings"].clear()
            state["hysteresis_counter"] = 0

        # If embeddings are disabled, simply append and bypass semantic drift checks
        if not self.use_embeddings:
            state["active_buffer"].append(turn_data)
            await self.store.save_state(tenant_id, user_id, state)
            return

        # B. SEMANTIC DRIFT CHECK
        if not state["buffer_embeddings"]:
            # Initial population
            state["active_buffer"].append(turn_data)
            state["buffer_embeddings"].append(embedding_vec)
            await self.store.save_state(tenant_id, user_id, state)
            return

        # Compute running mean vector (Centroid)
        centroid = self._compute_centroid(state["buffer_embeddings"])
        
        # Calculate Cosine Similarity
        similarity = self._cosine_similarity(embedding_vec, centroid)
        logger.debug(f"Computed semantic similarity: {similarity:.4f}")
        
        if similarity >= self.similarity_threshold:
            # User is on the same topic
            if state["hysteresis_counter"] > 0:
                logger.debug("Topic shift aborted. Resetting hysteresis.")
                
            state["hysteresis_counter"] = 0
            state["potential_shift_buffer"].clear()
            state["potential_shift_embeddings"].clear()
            
            state["active_buffer"].append(turn_data)
            state["buffer_embeddings"].append(embedding_vec)
            
        else:
            # Potential topic shift detected
            state["potential_shift_buffer"].append(turn_data)
            state["potential_shift_embeddings"].append(embedding_vec)
            state["hysteresis_counter"] += 1
            
            logger.debug(f"Potential topic shift detected. Hysteresis count: {state['hysteresis_counter']}/{self.hysteresis_threshold}")
            
            if state["hysteresis_counter"] >= self.hysteresis_threshold:
                # Topic shift confirmed
                logger.info("Topic shift confirmed. Flushing previous context.")
                await self._flush_to_worker(tenant_id, user_id, state)
                
                # Set the new topic baseline
                state["active_buffer"].extend(state["potential_shift_buffer"])
                state["buffer_embeddings"].extend(state["potential_shift_embeddings"])
                
                state["potential_shift_buffer"].clear()
                state["potential_shift_embeddings"].clear()
                state["hysteresis_counter"] = 0

        await self.store.save_state(tenant_id, user_id, state)
