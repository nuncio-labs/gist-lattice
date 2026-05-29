import asyncio
import os
from gistlattice import GistLattice

async def main():
    # Example 4: Environment Variables
    # Instead of passing arguments, we configure the entire app via OS environment variables.
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # We initialize with absolutely zero configuration!
    # GistLattice will automatically read the environment variables.
    memory = GistLattice(
        tenant_id="demo-tenant",
        user_id="david"
    )

    # print(f"Loaded Provider: {memory.service}")
    # print(f"Loaded Embedding: {memory.container.settings.embedding_provider}")
    # print(f"Loaded Queue: {memory.container.settings.queue_backend}")
    # print(f"Loaded Memory Limit: {memory.container.settings.memory_limit}")
    await memory.aremember(prompt="Hi, I am Saurav", response="Hi Saurav, how can I assist you?")
    result = await memory.ahydrate_context(prompt="Hi, I am Saurav")
    print(result)
    print("\nThe environment variables successfully configured the system globally!")

if __name__ == "__main__":
    asyncio.run(main())