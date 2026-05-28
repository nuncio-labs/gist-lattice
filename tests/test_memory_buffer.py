import unittest
import asyncio
from typing import List, Dict, Any

from gistlattice.memory_buffer import MemoryBufferController, InMemoryBufferStore

class MockWorker:
    """A mock worker to capture asynchronous flushes."""
    def __init__(self):
        self.flushed_payloads: List[List[Dict[str, Any]]] = []

    async def callback(self, tenant_id: str, user_id: str, payload: List[Dict[str, Any]]):
        self.flushed_payloads.append(payload)


class TestMemoryBufferController(unittest.IsolatedAsyncioTestCase):
    
    async def test_basic_append(self):
        mock_worker = MockWorker()
        store = InMemoryBufferStore()
        controller = MemoryBufferController(
            worker_callback=mock_worker.callback,
            store=store,
            max_messages=5,
            similarity_threshold=0.6,
            hysteresis_threshold=2,
            overlap_window_size=1,
            use_embeddings=True
        )
        
        await controller.process_turn("t1", "u1", "Hello", "Hi there", [1.0, 0.0])
        
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 1)
        self.assertEqual(state["active_buffer"][0]["prompt"], "Hello")
        self.assertEqual(len(mock_worker.flushed_payloads), 0)

    async def test_hard_limit_flush(self):
        mock_worker = MockWorker()
        store = InMemoryBufferStore()
        controller = MemoryBufferController(
            worker_callback=mock_worker.callback,
            store=store,
            max_messages=3,
            similarity_threshold=0.5,
            hysteresis_threshold=2,
            overlap_window_size=2,
            use_embeddings=True
        )
        
        vec = [1.0, 0.0]
        
        # Add 3 messages (doesn't flush yet, hits limit at the START of the 4th turn)
        await controller.process_turn("t1", "u1", "M1", "R1", vec)
        await controller.process_turn("t1", "u1", "M2", "R2", vec)
        await controller.process_turn("t1", "u1", "M3", "R3", vec)
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 3)
        
        # 4th message triggers hard limit flush
        await controller.process_turn("t1", "u1", "M4", "R4", vec)
        
        # Yield control to the event loop to allow the create_task(flush) to run
        await asyncio.sleep(0.01)
        
        self.assertEqual(len(mock_worker.flushed_payloads), 1)
        
        # Flushed payload should contain exactly M1, M2, M3
        flushed = mock_worker.flushed_payloads[0]
        self.assertEqual(len(flushed), 3)
        self.assertEqual(flushed[0]["prompt"], "M1")
        self.assertEqual(flushed[2]["prompt"], "M3")
        
        # After flush, overlap of 2 is retained (M2, M3), then M4 is appended
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 3)
        self.assertEqual(state["active_buffer"][0]["prompt"], "M2")
        self.assertEqual(state["active_buffer"][1]["prompt"], "M3")
        self.assertEqual(state["active_buffer"][2]["prompt"], "M4")

    async def test_semantic_drift_hysteresis(self):
        mock_worker = MockWorker()
        store = InMemoryBufferStore()
        controller = MemoryBufferController(
            worker_callback=mock_worker.callback,
            store=store,
            max_messages=10,
            similarity_threshold=0.6,
            hysteresis_threshold=2,
            overlap_window_size=1,
            use_embeddings=True
        )
        
        topic1_vec = [1.0, 0.0, 0.0]
        topic2_vec = [0.0, 1.0, 0.0]  # Orthogonal to topic1, cosine sim = 0.0
        
        # 1. Establish topic 1
        await controller.process_turn("t1", "u1", "T1_1", "R1", topic1_vec)
        await controller.process_turn("t1", "u1", "T1_2", "R2", topic1_vec)
        
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 2)
        self.assertEqual(len(state["potential_shift_buffer"]), 0)
        
        # 2. First off-topic message (Hysteresis = 1)
        await controller.process_turn("t1", "u1", "T2_1", "R3", topic2_vec)
        
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 2)
        self.assertEqual(len(state["potential_shift_buffer"]), 1)
        self.assertEqual(state["hysteresis_counter"], 1)
        self.assertEqual(len(mock_worker.flushed_payloads), 0)
        
        # 3. Second off-topic message confirms shift (Hysteresis = 2)
        await controller.process_turn("t1", "u1", "T2_2", "R4", topic2_vec)
        
        # Yield control to the event loop
        await asyncio.sleep(0.01)
        
        # Should have flushed the first topic
        self.assertEqual(len(mock_worker.flushed_payloads), 1)
        flushed = mock_worker.flushed_payloads[0]
        self.assertEqual(len(flushed), 2)
        self.assertEqual(flushed[0]["prompt"], "T1_1")
        
        # The active buffer should now have overlap=1 (T1_2) + the 2 shifted items (T2_1, T2_2)
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 3)
        self.assertEqual(state["active_buffer"][0]["prompt"], "T1_2")
        self.assertEqual(state["active_buffer"][1]["prompt"], "T2_1")
        self.assertEqual(state["active_buffer"][2]["prompt"], "T2_2")
        
        # Counters should be reset
        self.assertEqual(state["hysteresis_counter"], 0)
        self.assertEqual(len(state["potential_shift_buffer"]), 0)

    async def test_semantic_drift_aborted(self):
        mock_worker = MockWorker()
        store = InMemoryBufferStore()
        controller = MemoryBufferController(
            worker_callback=mock_worker.callback,
            store=store,
            max_messages=10,
            similarity_threshold=0.6,
            hysteresis_threshold=2,
            overlap_window_size=1,
            use_embeddings=True
        )
        
        topic1_vec = [1.0, 0.0, 0.0]
        topic2_vec = [0.0, 1.0, 0.0]
        
        # 1. Establish topic 1
        await controller.process_turn("t1", "u1", "T1_1", "R1", topic1_vec)
        
        # 2. Transient off-topic message (Hysteresis = 1)
        await controller.process_turn("t1", "u1", "T2_1", "R2", topic2_vec)
        state = await store.get_state("t1", "u1")
        self.assertEqual(state["hysteresis_counter"], 1)
        self.assertEqual(len(state["potential_shift_buffer"]), 1)
        
        # 3. User immediately returns to topic 1 (Hysteresis aborted)
        await controller.process_turn("t1", "u1", "T1_2", "R3", topic1_vec)
        
        # Hysteresis should be reset, transient shift dropped
        state = await store.get_state("t1", "u1")
        self.assertEqual(state["hysteresis_counter"], 0)
        self.assertEqual(len(state["potential_shift_buffer"]), 0)
        
        # Active buffer skips the transient topic, keeping only the continuous ones
        self.assertEqual(len(state["active_buffer"]), 2)
        self.assertEqual(state["active_buffer"][0]["prompt"], "T1_1")
        self.assertEqual(state["active_buffer"][1]["prompt"], "T1_2")
        
        self.assertEqual(len(mock_worker.flushed_payloads), 0)

    async def test_use_embeddings_false(self):
        mock_worker = MockWorker()
        store = InMemoryBufferStore()
        controller = MemoryBufferController(
            worker_callback=mock_worker.callback,
            store=store,
            max_messages=2,
            use_embeddings=False
        )
        
        # When use_embeddings=False, msg_embedding can be omitted
        await controller.process_turn("t1", "u1", "Hello", "Hi")
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 1)
        
        await controller.process_turn("t1", "u1", "World", "Greetings")
        
        await controller.process_turn("t1", "u1", "Third", "Turn")
        await asyncio.sleep(0.01)
        
        self.assertEqual(len(mock_worker.flushed_payloads), 1)
        self.assertEqual(len(mock_worker.flushed_payloads[0]), 2)
        self.assertEqual(mock_worker.flushed_payloads[0][0]["prompt"], "Hello")
        self.assertEqual(mock_worker.flushed_payloads[0][1]["prompt"], "World")
        
        state = await store.get_state("t1", "u1")
        self.assertEqual(len(state["active_buffer"]), 3)

if __name__ == '__main__':
    unittest.main()
