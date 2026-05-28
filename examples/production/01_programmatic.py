import asyncio
from gistlattice import GistLattice

async def main():
    # Example 3: Production Backends configured programmatically
    # We explicitly pass our Redis and Postgres configs via kwargs.
    
    memory = GistLattice(
        provider="openai",
        tenant_id="demo-tenant",
        user_id="charlie",
        
        # Enable the production backends
        queue_backend="redis",
        storage_backend="postgres",
        
        # Provide connection strings
        redis_url="redis://localhost:6379/0",
        postgres_url="postgresql://user:password@localhost:5432/gistlattice",
    )

    print("Configured GistLattice with Redis and Postgres!")
    print("Attempting to remember (requires actual running databases)...")
    
    try:
        analysis = await memory.remember(
            prompt="Let's build a new distributed system.",
            response="Sounds like a great technical challenge!"
        )
        print(f"Success! Memory Gist: {analysis.gist}")
    except Exception as e:
        print(f"\nCaught exception (expected if databases are not running locally): {e}")

if __name__ == "__main__":
    asyncio.run(main())
