# Production Backend Examples

These examples demonstrate how to configure GistLattice to connect to real, production-ready databases (Redis, Neo4j, and Qdrant) instead of the default in-memory stores.

## 1. Start the Databases

Before running these scripts, you must spin up the databases locally. We have provided a `docker-compose.yml` file to make this easy:

```bash
# Start Redis, Neo4j, and Qdrant in the background
docker compose up -d

# Wait a few seconds for the databases to initialize before running the scripts!
```

## 2. Run the Examples

Now that the databases are running, you can test the code:

```bash
# Example 1: Passing the URIs and credentials directly into the client
python 01_programmatic.py

# Example 2: Letting the client read the URIs from environment variables
python 02_environment_variables.py
```

## 3. Cleanup

When you are done testing, you can tear down the databases:

```bash
docker compose down
```
