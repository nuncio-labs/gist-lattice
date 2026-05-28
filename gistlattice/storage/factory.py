from gistlattice.config import Settings
from gistlattice.storage.base import StorageProvider

def get_storage_provider(settings: Settings) -> StorageProvider:
    if settings.storage_backend == "memory":
        from gistlattice.storage.memory import InMemoryStorageProvider
        return InMemoryStorageProvider()
    elif settings.storage_backend == "postgres":
        from gistlattice.storage.postgres import PostgresStorageProvider
        return PostgresStorageProvider(postgres_url=settings.postgres_url)
    elif settings.storage_backend == "neo4j":
        from gistlattice.storage.neo4j_provider import Neo4jStorageProvider
        return Neo4jStorageProvider(
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password
        )
    else:
        raise ValueError(f"Unknown storage_backend: {settings.storage_backend}")
