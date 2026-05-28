import asyncio
from gistlattice import GistLattice

async def main():
    # Example 2: Hybrid Gemini Async
    # We use Gemini for text generation, but OpenAI for embeddings.
    memory = GistLattice(
        provider="gemini",
        embedding_provider="openai",
        queue_backend="memory", # Even in-memory, queueing works asynchronously!
        tenant_id="demo-tenant",
        user_id="bob"
    )

    print("1. Queueing memory asynchronously...")
    # By setting run_in_background=True, the system instantly queues the 
    # interaction and returns a job ID. The actual LLM analysis runs in 
    # the background without blocking this thread.
    job_id = await memory.aremember(
        prompt="I'm feeling a bit anxious about the upcoming product launch.",
        response="It's completely normal to feel anxious. Let's review the checklist.",
        run_in_background=True,
        bypass_buffer=True
    )
    
    print(f"Instantly returned Job ID: {job_id}")
    print("The system is now processing this memory in the background.")
    
    # In a real app, you would return a 200 OK to the user right now.
    # We wait briefly here just to let the background worker finish.
    await asyncio.sleep(2)
    
    print("\n2. Hydrating Context for next prompt...")
    context = await memory.ahydrate_context("How am I feeling about work?")
    print(context)

if __name__ == "__main__":
    asyncio.run(main())
