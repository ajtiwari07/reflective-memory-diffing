# Reflective Memory Diffing Agent

A Python app that uses **Microsoft Agent Framework** with a **Redis provider** and **Redis chat message store** to drive a reflective memory diffing assistant. The agent can compare two memory snapshot files and explain changes.

## What The Agent Does

The Reflective Memory Diffing Agent helps keep long-term memory reliable over time.
It compares memory snapshots, detects drift, and applies repair actions to reduce stale or hallucinated memory.

Core capabilities:

- Tracks memory changes between snapshots (`added`, `removed`, `changed`).
- Detects drift types (`contradiction`, `staleness`, `redundancy`, `hallucination`, `preference_update`).
- Runs reflective prompting to choose repair actions (`rewrite`, `delete`, `annotate`, `keep`).
- Uses optional RAG sources (Bing or Azure AI Search) during repair for external grounding.
- Supports automatic snapshotting from conversation turns and optional auto-repair.

## Requirements

- Python 3.10+
- Microsoft Agent Framework (preview packages)
- Redis (local or Azure Managed Redis)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --pre
```

Create a `.env` file from `config/.env.example` and fill in values.

## Run

```powershell
python app.py
```

## LLM Configuration

This app supports:
- Microsoft Foundry (default)
- Azure OpenAI (chat client)

Set one of the following:

Foundry:
```
FOUNDRY_PROJECT_ENDPOINT=...
FOUNDRY_MODEL_DEPLOYMENT=gpt-4o-mini
```

Azure OpenAI:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

Bing (optional, used for RAG repairs):
```
BING_SEARCH_API_KEY=...
BING_SEARCH_ENDPOINT=https://api.bing.microsoft.com/v7.0/search
RAG_LOG_PATH=logs/rmd_rag.log
RAG_LOG_LEVEL=INFO
RAG_SOURCE=bing
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=...
AZURE_SEARCH_INDEX=your-index
AZURE_SEARCH_API_VERSION=2024-07-01
```

## Memory Snapshot Format

JSON with semantic memory entries (from the doc). Each snapshot is a list of memory items with metadata:

```json
{
  "snapshot_id": "M_t",
  "timestamp": "2026-02-12T00:00:00Z",
  "entries": [
    {
      "id": "e1",
      "content": "The capital of Peru is Lima.",
      "source": "external",
      "confidence": 0.92,
      "timestamp": "2026-02-01T00:00:00Z",
      "tags": ["geography"],
      "metadata": {"provenance": "wikipedia"},
      "embedding": [0.01, 0.02, 0.03]
    }
  ]
}
```

The JSON schema is defined in `snapshot_schema.json` and validated on load/store.

Drift detection includes: `added`, `removed`, `contradiction`, `staleness`, `redundancy`, and `hallucination`
(low-confidence LLM entries without support). Hallucinations are repaired by rewriting with context or deleted
if they cannot be repaired.

## Redis Configuration

Priority order:

1. `REDIS_URL` (full connection string)
2. `AZURE_REDIS_HOST` + `AZURE_REDIS_ACCESS_KEY`
3. Local default: `redis://localhost:6379`

For Azure Managed Redis, set:

```
AZURE_REDIS_HOST=your-cache-name.redis.cache.windows.net
AZURE_REDIS_ACCESS_KEY=your-access-key
AZURE_REDIS_PORT=10000
```

`AZURE_REDIS_PORT` defaults to `10000` to match Azure Managed Redis defaults; override if your cache uses `6380`.

Chat history replay behavior:

- `REUSE_CHAT_THREAD_ON_START=false` (default): start with a fresh chat thread each app run.
- `REUSE_CHAT_THREAD_ON_START=true`: reuse `THREAD_ID` and prior chat history from Redis.

## Snapshot Storage in Redis

Snapshots can be stored and retrieved by id. Keys use:

```
SNAPSHOT_KEY_PREFIX=rmd:snapshot
```

Tool usage (in chat):
- `store_snapshot(snapshot_id, snapshot_path)`
- `diff_snapshots(old_id, new_id)`
- `repair_snapshot_llm(old_id, new_id, repaired_id)`

`repair_snapshot_llm` uses Azure OpenAI plus a lightweight RAG context (top similar entries). A reflective
prompting step classifies drift and recommends an action (rewrite/delete/annotate/keep), which is recorded
in metadata and used to guide repairs. If Bing/Azure Search is configured, the top snippets are added as
external sources for repairs.

## Layout

- `src/reflective_memory_diffing_agent/`: package source code
- `config/`: `.env` and `.env.example`
- `logs/`: runtime logs (RAG)
- `snapshots/`: sample snapshots

## Auto Snapshotting

You can enable automatic memory snapshots from conversation turns:

```
AUTO_SNAPSHOT_INTERVAL=3
AUTO_SNAPSHOT_MAX_ENTRIES=200
AUTO_REPAIR_ON_SNAPSHOT=true
AUTO_SNAPSHOT_BACKGROUND=true
```

When enabled, the agent will store snapshots in Redis every N turns and optionally
run drift repair between the last snapshot and the new one.
With `AUTO_SNAPSHOT_BACKGROUND=true`, snapshot+repair jobs are queued in a background
worker to reduce chat latency. Preference queries flush pending jobs before lookup
to keep latest-value consistency.

## Preference Normalization (Hybrid)

Preference extraction/query routing supports three modes:

```
PREFERENCE_NORMALIZER_MODE=hybrid
PREFERENCE_NORMALIZER_MAX_MS=1200
```

- `deterministic`: regex-only normalization (fastest).
- `hybrid`: deterministic first, LLM fallback when regex misses.
- `llm`: LLM-only preference parsing.

Latency is measured for LLM normalization calls and written to `logs/rmd_rag.log`.
If a call exceeds `PREFERENCE_NORMALIZER_MAX_MS`, the LLM result is ignored for that turn.

## Hallucination Benchmark

An offline benchmark is available to quantify hallucination drift reduction:

```powershell
$env:PYTHONPATH="src"
python tests/evals/run_hallucination_benchmark.py
```

Details and case format: `tests/evals/README.md`.

## Embeddings & Vector Search

You can generate embeddings for snapshot entries and use Redis vector search
for faster semantic matching between snapshots.

```
EMBEDDING_PROVIDER=azure_openai
EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-10-21
USE_REDIS_VECTOR_SEARCH=true
REDIS_VECTOR_INDEX=rmd_entries
REDIS_VECTOR_KEY_PREFIX=rmd:entry
REDIS_VECTOR_DIM=
REDIS_VECTOR_DISTANCE=COSINE
```

When embeddings are present, snapshots are indexed in Redis using RediSearch.
Drift detection can then use KNN search (filtered by snapshot_id) to find the best
match from the prior snapshot.

## Tests

```powershell
pytest -q
```
