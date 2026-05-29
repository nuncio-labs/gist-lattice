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
This example loads configuration automatically via a `.env` file instead of explicit parameters.

1. Ensure you have the `python-dotenv` package installed so the script can load the `.env` file:
   ```bash
   pip install python-dotenv
   ```

2. Make sure you have your `OPENAI_API_KEY` set in the `.env` file, and then run the example:
   ```bash
   python 02_environment_variables.py
   ```
   *Note: Because `GISTLATTICE_BUFFER_SIZE=5` is set in the `.env` file, the interaction will be placed into a temporary Redis buffer. The buffer will only flush to the main queue once 5 interactions are stored! You can either run the script 5 times, or change the buffer size to `1` in the `.env` file.*

3. Start the background consolidation worker in a separate terminal to process the queue once the buffer flushes:
   ```bash
   python run_worker.py
   ```
```

## 3. Cleanup

When you are done testing, you can tear down the databases:

```bash
docker compose down
```
