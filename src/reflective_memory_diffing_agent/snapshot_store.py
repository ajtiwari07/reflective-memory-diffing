from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict

import redis
from jsonschema import Draft202012Validator


VECTOR_INDEX_NAME = os.getenv("REDIS_VECTOR_INDEX", "rmd_entries")
VECTOR_KEY_PREFIX = os.getenv("REDIS_VECTOR_KEY_PREFIX", "rmd:entry")
VECTOR_DIM_ENV = os.getenv("REDIS_VECTOR_DIM", "").strip()
VECTOR_DISTANCE = os.getenv("REDIS_VECTOR_DISTANCE", "COSINE").upper()


SCHEMA_PATH = Path(__file__).parent / "snapshot_schema.json"


def _snapshot_key(snapshot_id: str) -> str:
    prefix = os.getenv("SNAPSHOT_KEY_PREFIX", "rmd:snapshot")
    return f"{prefix}:{snapshot_id}"


def _get_client(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, decode_responses=True)


def _vector_key(snapshot_id: str, entry_id: str) -> str:
    return f"{VECTOR_KEY_PREFIX}:{snapshot_id}:{entry_id}"


def _pack_vector(values: Any) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _ensure_vector_index(client: redis.Redis, dim: int) -> None:
    if dim <= 0:
        return
    try:
        client.ft(VECTOR_INDEX_NAME).info()
        return
    except Exception:
        pass
    schema = [
        redis.commands.search.field.TagField("snapshot_id"),
        redis.commands.search.field.TagField("entry_id"),
        redis.commands.search.field.TextField("content"),
        redis.commands.search.field.TagField("source"),
        redis.commands.search.field.NumericField("confidence"),
        redis.commands.search.field.TextField("timestamp"),
        redis.commands.search.field.VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": VECTOR_DISTANCE,
            },
        ),
    ]
    try:
        from redis.commands.search.index_definition import IndexDefinition
    except Exception:
        from redis.commands.search.indexDefinition import IndexDefinition  # type: ignore
    client.ft(VECTOR_INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=[f"{VECTOR_KEY_PREFIX}:"]),
    )


def _upsert_vector_entries(client: redis.Redis, snapshot_id: str, entries: Any) -> None:
    dim_override = int(VECTOR_DIM_ENV) if VECTOR_DIM_ENV else None
    for entry in entries:
        embedding = entry.get("embedding")
        if not embedding:
            continue
        dim = dim_override or len(embedding)
        _ensure_vector_index(client, dim)
        key = _vector_key(snapshot_id, entry.get("id") or entry.get("entry_id") or "")
        payload = {
            "snapshot_id": snapshot_id,
            "entry_id": entry.get("id") or entry.get("entry_id") or "",
            "content": entry.get("content", ""),
            "source": entry.get("source", ""),
            "confidence": float(entry.get("confidence", 0.0)),
            "timestamp": entry.get("timestamp", ""),
            "embedding": _pack_vector(embedding),
        }
        client.hset(key, mapping=payload)


def _load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_snapshot_payload(payload: Dict[str, Any]) -> None:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        messages = "; ".join([f"{'.'.join([str(p) for p in e.path])}: {e.message}" for e in errors])
        raise ValueError(f"Invalid snapshot payload: {messages}")


def store_snapshot_json(snapshot_id: str, snapshot_path: str, redis_url: str) -> str:
    payload = Path(snapshot_path).read_text(encoding="utf-8")
    parsed = json.loads(payload)
    if not parsed.get("snapshot_id"):
        parsed["snapshot_id"] = snapshot_id
        payload = json.dumps(parsed, ensure_ascii=True)
    validate_snapshot_payload(parsed)
    key = _snapshot_key(snapshot_id)
    client = _get_client(redis_url)
    client.set(key, payload)
    _upsert_vector_entries(client, snapshot_id, parsed.get("entries", []))
    return f"Stored snapshot '{snapshot_id}' in Redis key '{key}'."


def store_snapshot_payload(snapshot_id: str, payload: Dict[str, Any], redis_url: str) -> str:
    if not payload.get("snapshot_id"):
        payload["snapshot_id"] = snapshot_id
    validate_snapshot_payload(payload)
    key = _snapshot_key(snapshot_id)
    client = _get_client(redis_url)
    client.set(key, json.dumps(payload, ensure_ascii=True))
    _upsert_vector_entries(client, snapshot_id, payload.get("entries", []))
    return f"Stored snapshot '{snapshot_id}' in Redis key '{key}'."


def load_snapshot_json(snapshot_id: str, redis_url: str) -> Dict[str, Any]:
    key = _snapshot_key(snapshot_id)
    client = _get_client(redis_url)
    payload = client.get(key)
    if payload is None:
        raise ValueError(f"Snapshot '{snapshot_id}' not found in Redis key '{key}'.")
    parsed = json.loads(payload)
    return parsed
