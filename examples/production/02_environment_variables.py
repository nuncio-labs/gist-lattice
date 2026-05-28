import asyncio
import os
from gistlattice import GistLattice

async def main():
    # Example 4: Environment Variables
    # Instead of passing arguments, we configure the entire app via OS environment variables.
    
    os.environ["GISTLATTICE_LLM_PROVIDER"] = "anthropic"
    os.environ["GISTLATTICE_EMBEDDING_PROVIDER"] = "openai"
    os.environ["GISTLATTICE_QUEUE_BACKEND"] = "redis"
    os.environ["GISTLATTICE_STORAGE_BACKEND"] = "postgres"
    os.environ["GISTLATTICE_POSTGRES_URL"] = "postgresql://user:password@localhost:5432/gistlattice"
    os.environ["GISTLATTICE_REDIS_URL"] = "redis://localhost:6379/1"
    os.environ["GISTLATTICE_MEMORY_LIMIT"] = "10"
    
    # We initialize with absolutely zero configuration!
    # GistLattice will automatically read the environment variables.
    memory = GistLattice(
        tenant_id="demo-tenant",
        user_id="david"
    )

    print(f"Loaded Provider: {memory.container.settings.llm_provider}")
    print(f"Loaded Embedding: {memory.container.settings.embedding_provider}")
    print(f"Loaded Queue: {memory.container.settings.queue_backend}")
    print(f"Loaded Memory Limit: {memory.container.settings.memory_limit}")
    
    print("\nThe environment variables successfully configured the system globally!")

if __name__ == "__main__":
    asyncio.run(main())
