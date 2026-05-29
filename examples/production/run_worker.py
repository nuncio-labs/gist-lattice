import asyncio
from dotenv import load_dotenv
import gistlattice.worker

if __name__ == "__main__":
    # Load environment variables from the .env file in this directory
    load_dotenv()
    
    print("Starting GistLattice Consolidation Worker...")
    gistlattice.worker.main()
