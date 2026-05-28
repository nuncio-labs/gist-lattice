import asyncio
from gistlattice import GistLattice

class SimpleAgent:
    def __init__(self):
        # 1. LONG-TERM MEMORY (GistLattice)
        # Using memory queue backend to simulate async processing
        self.long_term_memory = GistLattice(
            provider="openai", 
            queue_backend="memory",
            tenant_id="demo-tenant",
            user_id="user-123"
        )
        
        # 2. SHORT-TERM MEMORY (Sliding Window Buffer)
        # Keeps the exact literal transcript of the last 4 messages
        self.short_term_buffer = []
        self.MAX_BUFFER_SIZE = 4

    async def chat(self, user_message: str):
        print(f"\n[User]: {user_message}")
        
        # --- A. HYDRATE LONG-TERM CONTEXT ---
        # Fetch relevant past facts, emotional state, and active projects
        system_context = await self.long_term_memory.ahydrate_context(user_message)
        
        # --- B. CONSTRUCT FINAL PROMPT ---
        # Combine GistLattice (Long-Term) + Buffer (Short-Term) + Current Message
        print("\n[System internal prompt sent to LLM]:")
        print(f"--- SYSTEM ---\n{system_context}\n--------------")
        print(f"Short-Term Buffer: {self.short_term_buffer}")
        print(f"Current Message: {user_message}")
        
        # (Imagine we call the actual OpenAI ChatCompletion API here)
        agent_response = f"I am acknowledging: {user_message}"
        print(f"\n[Agent]: {agent_response}")
        
        # --- C. UPDATE SHORT-TERM BUFFER ---
        self.short_term_buffer.append({"role": "user", "content": user_message})
        self.short_term_buffer.append({"role": "assistant", "content": agent_response})
        
        # Truncate buffer to prevent context window bloat
        if len(self.short_term_buffer) > self.MAX_BUFFER_SIZE:
            self.short_term_buffer = self.short_term_buffer[-self.MAX_BUFFER_SIZE:]
            
        # --- D. ASYNCHRONOUSLY SAVE TO LONG-TERM MEMORY ---
        # We fire and forget! The user doesn't wait for this.
        job_id = await self.long_term_memory.aremember(
            prompt=user_message,
            response=agent_response,
            run_in_background=True
        )
        print(f"[System]: Background memory consolidation queued (Job ID: {job_id})")

async def main():
    agent = SimpleAgent()
    
    # Message 1: Will be stored asynchronously
    await agent.chat("Hi, my name is Alice and I am a software engineer.")
    
    # Wait a moment for the background worker to finish processing Message 1
    await asyncio.sleep(2) 
    
    # Message 2: Hydration will pull Alice's name from long-term memory!
    await agent.chat("Can you remind me what my job is?")

if __name__ == "__main__":
    asyncio.run(main())
