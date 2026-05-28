import asyncio
from gistlattice import GistLattice

async def main():
    # Example 3: Production Backends configured programmatically
    # We explicitly pass our Redis, Neo4j, and Qdrant config via kwargs.
    
    memory = GistLattice(
        provider="openai",
        tenant_id="demo-tenant",
        user_id="charlie",
        
        # Enable the production backends
        queue_backend="redis",
        semantic_store_backend="neo4j",
        episodic_store_backend="qdrant",
        
        # Provide connection strings
        redis_url="redis://localhost:6379/0",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        qdrant_host="localhost",
        qdrant_port=6333,
    )

    print("Configured GistLattice with Redis, Neo4j, and Qdrant!")
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
