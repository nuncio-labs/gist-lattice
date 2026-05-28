from abc import ABC, abstractmethod
from typing import Any
from gistlattice.models import ExtractedMemory, MemoryGist

class StorageProvider(ABC):
    """
    Unified abstract base class for Pluggable Storage Providers.
    Handles both vector-based episodic memories and graph-based semantic entity relationships.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Sets up database connections, creates schema tables/nodes, and builds native vector indices (HNSW/Cosine).
        """
        pass

    @abstractmethod
    async def write_memory(self, memory: ExtractedMemory) -> None:
        """
        Persists gists, vectors, entities, and relationships atomically inside a transaction.
        """
        pass

    @abstractmethod
    async def vector_search(self, user_id: str, tenant_id: str, query_vector: list[float], limit: int) -> list[MemoryGist]:
        """
        Performs multi-tenant similarity retrieval using native vector indices.
        Returns a list of `MemoryGist` objects representing the closest episodic memories.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes any underlying database connection pools.
        """
        pass
