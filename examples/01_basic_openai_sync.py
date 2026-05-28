import asyncio
from gistlattice import GistLattice

async def main():
    # Example 1: Basic OpenAI with synchronous execution
    # We explicitly specify the LLM model and Embedding model here.
    memory = GistLattice(
        provider="openai",
        llm_model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
        tenant_id="demo-tenant",
        user_id="alice"
    )

    print("1. Saving memory synchronously...")
    # Because run_in_background=False (which is the default), this will 
    # block and wait for the LLM to analyze and save the memory.
    analysis = await memory.aremember(
        prompt="I really need to focus on my Python project this weekend.",
        response="Got it, I'll remind you to work on your Python project.",
        bypass_buffer=True
    )
    
    print("\nMemory Analysis Results:")
    print(f"- Gist: {analysis.gist}")
    print(f"- Valence: {analysis.valence}")
    print(f"- Importance: {analysis.importance}")

    print("\n2. Hydrating Context for next prompt...")
    context = await memory.ahydrate_context("What was I planning to do?")
    print(context)

if __name__ == "__main__":
    asyncio.run(main())
