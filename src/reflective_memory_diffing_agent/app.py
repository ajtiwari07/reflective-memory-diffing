from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4

from agent_framework.azure import AzureAIClient, AzureOpenAIChatClient
from agent_framework.redis import RedisChatMessageStore, RedisProvider
from agent_framework.exceptions import ServiceInvalidRequestError
from azure.identity.aio import DefaultAzureCredential
import aiohttp

from .config import (
    load_azure_search_config,
    load_bing_config,
    load_embedding_config,
    load_llm_config,
    load_rag_config,
    load_redis_config,
)
from .memory_diff import diff_snapshots_by_id, diff_snapshots_from_files, repair_snapshot_by_id, repair_snapshot_with_rewriter
from .snapshot_store import store_snapshot_json, store_snapshot_payload


class ResilientRedisProvider(RedisProvider):
    """RedisProvider with error handling for search failures."""
    
    async def invoking(self, messages, **kwargs):
        """Override invoking to catch search syntax errors."""
        try:
            return await super().invoking(messages, **kwargs)
        except ServiceInvalidRequestError as e:
            if "Syntax error" in str(e) or "Redis text search failed" in str(e):
                # Log the error and return empty context rather than crashing
                print(f"\n[Warning: Redis search syntax error: {e}. Continuing without context.]")
                return ""
            raise


async def create_provider() -> RedisProvider:
    config = load_redis_config()
    return ResilientRedisProvider(
        redis_url=config.redis_url,
        index_name="reflective_memory_diffing",
        prefix="rmd",
        application_id=config.application_id,
        agent_id=config.agent_id,
        user_id=config.user_id,
    )


def create_chat_store_factory(thread_id_holder: dict, redis_url: str):
    def factory() -> RedisChatMessageStore:
        return RedisChatMessageStore(
            redis_url=redis_url,
            thread_id=thread_id_holder["id"],
            key_prefix="rmd_chat",
            max_messages=100,
        )

    return factory


async def main() -> None:
    log_path = os.getenv("RAG_LOG_PATH", "logs/rmd_rag.log")
    if not os.path.isabs(log_path):
        log_path = str(Path(__file__).resolve().parents[2] / log_path)
    log_level = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    llm_config = load_llm_config()
    bing_config = load_bing_config()
    azure_search_config = load_azure_search_config()
    rag_config = load_rag_config()
    embedding_config = load_embedding_config()
    auto_snapshot_interval = int(os.getenv("AUTO_SNAPSHOT_INTERVAL", "0"))
    auto_snapshot_max = int(os.getenv("AUTO_SNAPSHOT_MAX_ENTRIES", "200"))
    auto_repair = os.getenv("AUTO_REPAIR_ON_SNAPSHOT", "false").strip().lower() in {"1", "true", "yes"}
    auto_snapshot_background = os.getenv("AUTO_SNAPSHOT_BACKGROUND", "false").strip().lower() in {"1", "true", "yes"}
    logging.info(
        "RAG config: source=%s azure_search_endpoint=%s azure_search_index=%s",
        rag_config.source,
        azure_search_config.endpoint,
        azure_search_config.index_name,
    )
    logging.info(
        "Auto snapshot: interval=%s max_entries=%s auto_repair=%s background=%s",
        auto_snapshot_interval,
        auto_snapshot_max,
        auto_repair,
        auto_snapshot_background,
    )
    logging.info(
        "Embedding config: provider=%s model=%s",
        embedding_config.provider,
        embedding_config.model,
    )
    if llm_config.provider == "foundry" and not llm_config.foundry_endpoint:
        raise ValueError("FOUNDRY_PROJECT_ENDPOINT is required for Foundry mode")
    preference_pattern = re.compile(
        r"\b(?:my|i)\s+(?:favorite|fav|favourite)\s+([a-zA-Z ]{2,50})\s+is\s+([a-zA-Z ]{2,50})\b",
        re.IGNORECASE,
    )

    def _normalize_preference_key(raw_key: str) -> str:
        key = " ".join(raw_key.strip().lower().split())
        key = key.replace("favourite", "favorite")
        key = re.sub(r"\bmy\b", "", key).strip()
        key = re.sub(r"\bfood\b", "cuisine", key)
        key = re.sub(r"\bfav\b", "favorite", key)
        key = re.sub(r"\bcuisines\b", "cuisine", key)
        key = re.sub(r"\s+", " ", key).strip()
        if key in {"cuisine", "food"}:
            key = "cuisine"
        if "cuisine" in key or "food" in key:
            return "favorite_cuisine"
        return "_".join(key.split())

    def _extract_preferences(text: str) -> List[Dict[str, Any]]:
        matches = preference_pattern.findall(text or "")
        entries = []
        for key, value in matches:
            key_norm = _normalize_preference_key(key)
            value_norm = " ".join(value.strip().split())
            entries.append(
                {
                    "content": f"User favorite {key_norm.replace('_', ' ')} is {value_norm}.",
                    "source": "user",
                    "confidence": 0.95,
                    "metadata": {
                        "preference_key": key_norm,
                        "preference_value": value_norm,
                        "preference_subject": "user",
                    },
                }
            )
        return entries

    preference_query_pattern = re.compile(
        r"\b(?:what\s+is|tell\s+me)\s+(?:my|the)\s+(?:favorite|fav|favourite)\s+([a-zA-Z ]{1,40})\??\b",
        re.IGNORECASE,
    )

    def _extract_preference_query_key(text: str) -> Optional[str]:
        match = preference_query_pattern.search(text or "")
        if not match:
            return None
        raw = match.group(1).strip()
        if not raw:
            return None
        return _normalize_preference_key(raw)

    def _parse_iso_ts(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def get_latest_preference(preference_key: str = "favorite_cuisine") -> str:
        try:
            import redis
        except Exception as exc:
            return f"Preference lookup unavailable: {exc}"

        key_norm = _normalize_preference_key(preference_key)
        snapshot_prefix = os.getenv("SNAPSHOT_KEY_PREFIX", "rmd:snapshot")
        client = redis.Redis.from_url(redis_config.redis_url, decode_responses=True)

        cursor = 0
        latest_entry = None
        latest_ts = None
        pattern = f"{snapshot_prefix}:auto_*"
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=500)
            for redis_key in keys:
                raw = client.get(redis_key)
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for entry in payload.get("entries", []):
                    metadata = entry.get("metadata", {}) or {}
                    if metadata.get("preference_key") != key_norm:
                        continue
                    ts = _parse_iso_ts(entry.get("timestamp")) or _parse_iso_ts(payload.get("timestamp"))
                    if ts is None:
                        ts = datetime.min.replace(tzinfo=timezone.utc)
                    if latest_ts is None or ts > latest_ts:
                        latest_ts = ts
                        latest_entry = entry
            if cursor == 0:
                break

        if not latest_entry:
            return f"No preference found for key '{key_norm}'."
        value = latest_entry.get("metadata", {}).get("preference_value")
        if not value:
            # Fallback to content if metadata value is missing
            return f"Latest preference for '{key_norm}': {latest_entry.get('content', '')}"
        return f"Latest preference for '{key_norm}': {value}"

    async def _embed_texts(texts: List[str]) -> List[List[float]]:
        if embedding_config.provider in {"none", "", "off"}:
            logging.info("Embeddings skipped: provider=%s", embedding_config.provider)
            return [[] for _ in texts]
        if embedding_config.provider == "azure_openai":
            try:
                from openai import AzureOpenAI
            except Exception:
                logging.exception("Embeddings unavailable: openai package not found")
                return [[] for _ in texts]
            logging.info(
                "Embedding request: provider=azure_openai model=%s count=%s",
                embedding_config.model,
                len(texts),
            )
            def _embed_sync():
                client = AzureOpenAI(
                    api_key=embedding_config.azure_openai_api_key,
                    azure_endpoint=embedding_config.azure_openai_endpoint,
                    api_version=embedding_config.azure_openai_api_version,
                )
                return client.embeddings.create(model=embedding_config.model, input=texts)

            resp = await asyncio.to_thread(_embed_sync)
            return [d.embedding for d in resp.data]
        logging.info("Embeddings skipped: unsupported provider=%s", embedding_config.provider)
        return [[] for _ in texts]

    def _make_entry(content: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "id": uuid4().hex,
            "content": content,
            "source": source,
            "confidence": 0.7 if source == "assistant" else 0.95,
            "timestamp": now,
            "metadata": metadata or {},
        }

    async def _auto_snapshot(
        memory_entries: List[Dict[str, Any]],
        last_snapshot_id: Optional[str],
        store_payload_fn,
        repair_fn,
    ) -> Optional[str]:
        if auto_snapshot_interval <= 0:
            return last_snapshot_id
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        snapshot_id = f"auto_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        trimmed = memory_entries[-auto_snapshot_max:] if auto_snapshot_max > 0 else memory_entries
        texts = [e.get("content", "") for e in trimmed]
        embeddings = await _embed_texts(texts)
        for entry, emb in zip(trimmed, embeddings):
            if emb:
                entry["embedding"] = emb
        logging.info(
            "Auto snapshot embeddings: entries=%s embedded=%s",
            len(trimmed),
            sum(1 for e in trimmed if e.get("embedding")),
        )
        payload = {
            "snapshot_id": snapshot_id,
            "timestamp": now,
            "entries": trimmed,
            "metadata": {"auto_snapshot": True},
        }
        await asyncio.to_thread(store_payload_fn, snapshot_id, payload, redis_config.redis_url)
        logging.info("Auto snapshot stored: %s entries=%s", snapshot_id, len(trimmed))
        if auto_repair and last_snapshot_id:
            repaired_id = f"{snapshot_id}_repaired"
            repaired = await repair_fn(last_snapshot_id, snapshot_id, repaired_id)
            logging.info("Auto repair stored: %s", repaired_id)
            return repaired_id
        return snapshot_id

    def _clone_entries(memory_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Create an immutable snapshot payload for background processing.
        return json.loads(json.dumps(memory_entries))

    if llm_config.provider == "azure_openai":
        if not llm_config.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI mode")
        if not llm_config.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI mode")

    redis_config = load_redis_config()
    provider = await create_provider()

    thread_id_holder = {"id": redis_config.thread_id or f"memory_diff_thread_{uuid4().hex}"}
    def _new_chat_store_factory() -> RedisChatMessageStore:
        factory = create_chat_store_factory(thread_id_holder, redis_config.redis_url)
        return factory()

    async def _bing_search(query: str, top_k: int = 3) -> List[str]:
        if not bing_config.api_key:
            logging.info("Bing RAG skipped: missing BING_SEARCH_API_KEY")
            return []
        endpoint = bing_config.endpoint.rstrip("/")
        if not endpoint.endswith("/search"):
            endpoint = f"{endpoint}/search"
        headers = {"Ocp-Apim-Subscription-Key": bing_config.api_key}
        params = {"q": query, "mkt": "en-US", "count": top_k, "textFormat": "Raw"}
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        logging.warning("Bing RAG HTTP %s for query=%s", resp.status, query)
                        return []
                    payload = await resp.json()
        except Exception:
            logging.exception("Bing RAG request failed for query=%s", query)
            return []

        results = payload.get("webPages", {}).get("value", [])
        lines: List[str] = []
        for item in results:
            name = str(item.get("name", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not name and not snippet:
                continue
            text = f"{name}: {snippet}".strip(": ").strip()
            lines.append(f"- {text[:240]}")
        logging.info("Bing RAG results=%s query=%s", len(lines), query)
        return lines

    async def _azure_search(query: str, top_k: int = 3) -> List[str]:
        if not azure_search_config.endpoint or not azure_search_config.api_key or not azure_search_config.index_name:
            logging.info("Azure Search RAG skipped: missing AZURE_SEARCH_* configuration")
            return []
        endpoint = azure_search_config.endpoint.rstrip("/")
        url = (
            f"{endpoint}/indexes/{azure_search_config.index_name}/docs/search"
            f"?api-version={azure_search_config.api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": azure_search_config.api_key,
        }
        payload = {
            "search": query,
            "top": top_k,
            "queryType": "simple",
        }
        logging.info(
            "Azure Search request: endpoint=%s index=%s query=%s",
            azure_search_config.endpoint,
            azure_search_config.index_name,
            query,
        )
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        logging.warning("Azure Search RAG HTTP %s for query=%s", resp.status, query)
                        return []
                    data = await resp.json()
        except Exception:
            logging.exception("Azure Search RAG request failed for query=%s", query)
            return []

        results = data.get("value", [])
        lines: List[str] = []
        for item in results:
            # Prefer common content fields if present
            snippet = ""
            for key in ("content", "text", "chunk", "summary", "description"):
                if key in item and item[key]:
                    snippet = str(item[key]).strip()
                    break
            if not snippet:
                continue
            lines.append(f"- {snippet[:240]}")
        logging.info("Azure Search RAG results=%s query=%s", len(lines), query)
        return lines

    async def _rag_sources(query: str) -> List[str]:
        source = rag_config.source
        lines: List[str] = []
        if source in {"bing", "both"}:
            lines.extend(await _bing_search(query))
        if source in {"azure_search", "azure", "both"}:
            lines.extend(await _azure_search(query))
        if source in {"none", "off"}:
            return []
        return lines

    if llm_config.provider == "azure_openai":
        client = AzureOpenAIChatClient(
            endpoint=llm_config.azure_openai_endpoint,
            deployment_name=llm_config.azure_openai_deployment,
            api_key=llm_config.azure_openai_api_key,
        )

        def store_snapshot(snapshot_id: str, snapshot_path: str) -> str:
            return store_snapshot_json(snapshot_id, snapshot_path, redis_config.redis_url)

        def diff_snapshots(old_id: str, new_id: str) -> str:
            return diff_snapshots_by_id(old_id, new_id, redis_config.redis_url)

        def repair_snapshot(old_id: str, new_id: str, repaired_id: str) -> str:
            repaired = repair_snapshot_by_id(old_id, new_id, repaired_id, redis_config.redis_url)
            return store_snapshot_payload(repaired_id, repaired, redis_config.redis_url)

        async def _reflect_with_llm(entry, context, drift) -> Optional[Dict[str, Any]]:
            context_lines = []
            for c in context:
                context_lines.append(
                    f"- {c.content} (source={c.source}, confidence={c.confidence})"
                )

            prompt = (
                "You are performing reflective memory drift analysis.\n"
                "Given the target memory entry and context, assess drift and choose an action.\n"
                "Return ONLY valid JSON with keys: assessment, action, confidence, rationale.\n"
                "Valid actions: rewrite, delete, annotate, keep.\n\n"
                f"Drift type: {drift.drift_type}\n"
                f"Reason: {drift.reason}\n\n"
                f"Target entry:\n{entry.content}\n\n"
                "Context entries:\n"
                + "\n".join(context_lines)
            )

            reflect_agent = client.as_agent(
                name="MemoryReflectAgent",
                instructions=(
                    "You assess drift and choose an action. "
                    "Return strict JSON only."
                ),
            )
            response = await reflect_agent.run(prompt, thread=reflect_agent.get_new_thread())
            if not response or not response.text:
                return None
            text = response.text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        async def _rewrite_with_rag(entry, context, drift) -> str | None:
            context_lines = []
            for c in context:
                context_lines.append(
                    f"- {c.content} (source={c.source}, confidence={c.confidence})"
                )

            external_lines = await _rag_sources(entry.content)
            logging.info(
                "RAG rewrite entry_id=%s drift=%s external_sources=%s",
                entry.entry_id,
                drift.drift_type,
                len(external_lines),
            )
            prompt = (
                "You are repairing a memory entry in a long-term agent memory.\n"
                "Task: Rewrite the target entry to be accurate and consistent with the context.\n"
                "Return ONLY the corrected entry text, no extra words.\n\n"
                f"Drift type: {drift.drift_type}\n"
                f"Reason: {drift.reason}\n\n"
                f"Target entry:\n{entry.content}\n\n"
                "Context entries:\n"
                + "\n".join(context_lines)
                + ("\n\nExternal sources (Bing):\n" + "\n".join(external_lines) if external_lines else "")
            )

            repair_agent = client.as_agent(
                name="MemoryRepairAgent",
                instructions=(
                    "You rewrite a single memory entry using the provided context. "
                    "Return only the corrected entry text."
                ),
            )
            response = await repair_agent.run(prompt, thread=repair_agent.get_new_thread())
            if response and response.text:
                return response.text.strip().strip('\"')
            return None

        async def repair_snapshot_llm(old_id: str, new_id: str, repaired_id: str) -> str:
            repaired = await repair_snapshot_with_rewriter(
                old_id,
                new_id,
                repaired_id,
                redis_config.redis_url,
                _rewrite_with_rag,
                _reflect_with_llm,
            )
            return store_snapshot_payload(repaired_id, repaired, redis_config.redis_url)

        def _build_agent():
            return client.as_agent(
                name="ReflectiveMemoryDiffAgent",
                instructions=(
                    "You are a reflective memory diffing assistant. "
                    "When the user asks to compare two memory snapshots, "
                    "use the tool diff_snapshots_from_files with file paths. "
                    "If snapshots are stored in Redis by id, use store_snapshot "
                    "and diff_snapshots. If asked to repair drift, use repair_snapshot_llm. "
                    "When asked for user preferences (favorite cuisine, etc.), "
                    "first call get_latest_preference and answer from that result. "
                    "If no relevant memory is found, answer from general knowledge and "
                    "label it as such. Prefer memory when available."
                ),
                tools=[diff_snapshots_from_files, store_snapshot, diff_snapshots, repair_snapshot_llm, get_latest_preference],
                context_provider=provider,
                chat_message_store_factory=_new_chat_store_factory,
            )

        agent = _build_agent()
        thread = agent.get_new_thread()
        memory_entries: List[Dict[str, Any]] = []
        turn_count = 0
        snapshot_state: Dict[str, Optional[str]] = {"last_snapshot_id": None}
        snapshot_queue: asyncio.Queue[Optional[List[Dict[str, Any]]]] = asyncio.Queue()
        snapshot_worker_task: Optional[asyncio.Task] = None

        async def _run_snapshot_now(entries_snapshot: List[Dict[str, Any]]) -> None:
            snapshot_state["last_snapshot_id"] = await _auto_snapshot(
                entries_snapshot,
                snapshot_state["last_snapshot_id"],
                store_snapshot_payload,
                repair_snapshot_llm,
            )

        async def _snapshot_worker() -> None:
            while True:
                job = await snapshot_queue.get()
                try:
                    if job is None:
                        return
                    await _run_snapshot_now(job)
                except Exception:
                    logging.exception("Background auto-snapshot job failed")
                finally:
                    snapshot_queue.task_done()

        async def _schedule_snapshot_if_due() -> None:
            if auto_snapshot_interval <= 0 or (turn_count % auto_snapshot_interval) != 0:
                return
            entries_snapshot = _clone_entries(memory_entries)
            if auto_snapshot_background:
                await snapshot_queue.put(entries_snapshot)
            else:
                await _run_snapshot_now(entries_snapshot)

        async def _flush_snapshot_jobs() -> None:
            if auto_snapshot_background:
                await snapshot_queue.join()

        async def _stop_snapshot_worker() -> None:
            if snapshot_worker_task is not None:
                await snapshot_queue.put(None)
                await snapshot_queue.join()
                await snapshot_worker_task

        if auto_snapshot_background:
            snapshot_worker_task = asyncio.create_task(_snapshot_worker())

        print("Reflective Memory Diffing Agent (Azure OpenAI)")
        print("Type 'exit' to quit.\n")

        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    print("\nAgent: Input stream closed. Exiting.")
                    break
                if user_input.lower() in {"exit", "quit"}:
                    break
                if not user_input:
                    continue
                if user_input.lower() in {"reset", "/reset"}:
                    thread_id_holder["id"] = f"memory_diff_thread_{uuid4().hex}"
                    agent = _build_agent()
                    thread = agent.get_new_thread()
                    print("Agent: Thread reset.\n")
                    continue

                queried_pref_key = _extract_preference_query_key(user_input)
                if queried_pref_key:
                    await _flush_snapshot_jobs()
                    pref_result = get_latest_preference(queried_pref_key)
                    if pref_result.startswith("Latest preference for"):
                        value = pref_result.split(":", 1)[1].strip()
                        response_text = f"Based on your most recent memory, your {queried_pref_key.replace('_', ' ')} is {value}."
                    else:
                        response_text = pref_result
                    print(f"Agent: {response_text}\n")
                    memory_entries.append(_make_entry(user_input, "user"))
                    memory_entries.append(_make_entry(response_text, "assistant"))
                    turn_count += 1
                    await _schedule_snapshot_if_due()
                    continue

                print("Agent: ", end="", flush=True)
                try:
                    response = await agent.run(user_input, thread=thread)
                except Exception as exc:
                    if "role 'tool' must be a response to a preceeding message" in str(exc):
                        thread_id_holder["id"] = f"memory_diff_thread_{uuid4().hex}"
                        agent = _build_agent()
                        thread = agent.get_new_thread()
                        print("Agent: Thread was reset due to tool message ordering error. Please retry.\n")
                        continue
                    raise
                print(response.text)
                print()

                memory_entries.append(_make_entry(user_input, "user"))
                for pref in _extract_preferences(user_input):
                    memory_entries.append(_make_entry(pref["content"], pref["source"], pref["metadata"]))
                memory_entries.append(_make_entry(response.text or "", "assistant"))
                turn_count += 1
                await _schedule_snapshot_if_due()
        finally:
            await _stop_snapshot_worker()
    else:
        async with (
            DefaultAzureCredential() as credential,
            AzureAIClient(
                project_endpoint=llm_config.foundry_endpoint,
                model_deployment_name=llm_config.foundry_model,
                credential=credential,
            ) as client,
        ):
            def store_snapshot(snapshot_id: str, snapshot_path: str) -> str:
                return store_snapshot_json(snapshot_id, snapshot_path, redis_config.redis_url)

            def diff_snapshots(old_id: str, new_id: str) -> str:
                return diff_snapshots_by_id(old_id, new_id, redis_config.redis_url)

            async def _reflect_with_llm(entry, context, drift) -> Optional[Dict[str, Any]]:
                context_lines = []
                for c in context:
                    context_lines.append(
                        f"- {c.content} (source={c.source}, confidence={c.confidence})"
                    )

                prompt = (
                    "You are performing reflective memory drift analysis.\n"
                    "Given the target memory entry and context, assess drift and choose an action.\n"
                    "Return ONLY valid JSON with keys: assessment, action, confidence, rationale.\n"
                    "Valid actions: rewrite, delete, annotate, keep.\n\n"
                    f"Drift type: {drift.drift_type}\n"
                    f"Reason: {drift.reason}\n\n"
                    f"Target entry:\n{entry.content}\n\n"
                    "Context entries:\n"
                    + "\n".join(context_lines)
                )

                response = await agent.run(prompt, thread=agent.get_new_thread())
                if not response or not response.text:
                    return None
                text = response.text.strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None

            async def _rewrite_with_rag(entry, context, drift) -> str | None:
                context_lines = []
                for c in context:
                    context_lines.append(
                        f"- {c.content} (source={c.source}, confidence={c.confidence})"
                    )

                external_lines = await _rag_sources(entry.content)
                logging.info(
                    "RAG rewrite entry_id=%s drift=%s external_sources=%s",
                    entry.entry_id,
                    drift.drift_type,
                    len(external_lines),
                )
                prompt = (
                    "You are repairing a memory entry in a long-term agent memory.\n"
                    "Task: Rewrite the target entry to be accurate and consistent with the context.\n"
                    "Return ONLY the corrected entry text, no extra words.\n\n"
                    f"Drift type: {drift.drift_type}\n"
                    f"Reason: {drift.reason}\n\n"
                    f"Target entry:\n{entry.content}\n\n"
                    "Context entries:\n"
                    + "\n".join(context_lines)
                    + ("\n\nExternal sources (Bing):\n" + "\n".join(external_lines) if external_lines else "")
                )

                response = await agent.run(prompt, thread=agent.get_new_thread())
                if response and response.text:
                    return response.text.strip().strip('\"')
                return None

            async def repair_snapshot_llm(old_id: str, new_id: str, repaired_id: str) -> str:
                repaired = await repair_snapshot_with_rewriter(
                    old_id,
                    new_id,
                    repaired_id,
                    redis_config.redis_url,
                    _rewrite_with_rag,
                    _reflect_with_llm,
                )
                return store_snapshot_payload(repaired_id, repaired, redis_config.redis_url)

            def _build_agent():
                return client.as_agent(
                    name="ReflectiveMemoryDiffAgent",
                    instructions=(
                        "You are a reflective memory diffing assistant. "
                        "When the user asks to compare two memory snapshots, "
                        "use the tool diff_snapshots_from_files with file paths. "
                        "If snapshots are stored in Redis by id, use store_snapshot "
                        "and diff_snapshots. If asked to repair drift, use repair_snapshot_llm. "
                        "When asked for user preferences (favorite cuisine, etc.), "
                        "first call get_latest_preference and answer from that result. "
                        "If no relevant memory is found, answer from general knowledge and "
                        "label it as such. Prefer memory when available."
                    ),
                    tools=[diff_snapshots_from_files, store_snapshot, diff_snapshots, repair_snapshot_llm, get_latest_preference],
                    context_provider=provider,
                    chat_message_store_factory=_new_chat_store_factory,
                )

            agent = _build_agent()
            thread = agent.get_new_thread()
            memory_entries: List[Dict[str, Any]] = []
            turn_count = 0
            snapshot_state: Dict[str, Optional[str]] = {"last_snapshot_id": None}
            snapshot_queue: asyncio.Queue[Optional[List[Dict[str, Any]]]] = asyncio.Queue()
            snapshot_worker_task: Optional[asyncio.Task] = None

            async def _run_snapshot_now(entries_snapshot: List[Dict[str, Any]]) -> None:
                snapshot_state["last_snapshot_id"] = await _auto_snapshot(
                    entries_snapshot,
                    snapshot_state["last_snapshot_id"],
                    store_snapshot_payload,
                    repair_snapshot_llm,
                )

            async def _snapshot_worker() -> None:
                while True:
                    job = await snapshot_queue.get()
                    try:
                        if job is None:
                            return
                        await _run_snapshot_now(job)
                    except Exception:
                        logging.exception("Background auto-snapshot job failed")
                    finally:
                        snapshot_queue.task_done()

            async def _schedule_snapshot_if_due() -> None:
                if auto_snapshot_interval <= 0 or (turn_count % auto_snapshot_interval) != 0:
                    return
                entries_snapshot = _clone_entries(memory_entries)
                if auto_snapshot_background:
                    await snapshot_queue.put(entries_snapshot)
                else:
                    await _run_snapshot_now(entries_snapshot)

            async def _flush_snapshot_jobs() -> None:
                if auto_snapshot_background:
                    await snapshot_queue.join()

            async def _stop_snapshot_worker() -> None:
                if snapshot_worker_task is not None:
                    await snapshot_queue.put(None)
                    await snapshot_queue.join()
                    await snapshot_worker_task

            if auto_snapshot_background:
                snapshot_worker_task = asyncio.create_task(_snapshot_worker())

            print("Reflective Memory Diffing Agent (Foundry)")
            print("Type 'exit' to quit.\n")

            try:
                while True:
                    try:
                        user_input = input("You: ").strip()
                    except EOFError:
                        print("\nAgent: Input stream closed. Exiting.")
                        break
                    if user_input.lower() in {"exit", "quit"}:
                        break
                    if not user_input:
                        continue
                    if user_input.lower() in {"reset", "/reset"}:
                        thread_id_holder["id"] = f"memory_diff_thread_{uuid4().hex}"
                        agent = _build_agent()
                        thread = agent.get_new_thread()
                        print("Agent: Thread reset.\n")
                        continue

                    queried_pref_key = _extract_preference_query_key(user_input)
                    if queried_pref_key:
                        await _flush_snapshot_jobs()
                        pref_result = get_latest_preference(queried_pref_key)
                        if pref_result.startswith("Latest preference for"):
                            value = pref_result.split(":", 1)[1].strip()
                            response_text = f"Based on your most recent memory, your {queried_pref_key.replace('_', ' ')} is {value}."
                        else:
                            response_text = pref_result
                        print(f"Agent: {response_text}\n")
                        memory_entries.append(_make_entry(user_input, "user"))
                        memory_entries.append(_make_entry(response_text, "assistant"))
                        turn_count += 1
                        await _schedule_snapshot_if_due()
                        continue

                    print("Agent: ", end="", flush=True)
                    try:
                        response_text = ""
                        async for chunk in agent.run_stream(user_input, thread=thread):
                            if chunk.text:
                                response_text += chunk.text
                                print(chunk.text, end="", flush=True)
                        print("\n")
                    except Exception as exc:
                        if "role 'tool' must be a response to a preceeding message" in str(exc):
                            thread_id_holder["id"] = f"memory_diff_thread_{uuid4().hex}"
                            agent = _build_agent()
                            thread = agent.get_new_thread()
                            print("Agent: Thread was reset due to tool message ordering error. Please retry.\n")
                            continue
                        raise

                    memory_entries.append(_make_entry(user_input, "user"))
                    for pref in _extract_preferences(user_input):
                        memory_entries.append(_make_entry(pref["content"], pref["source"], pref["metadata"]))
                    memory_entries.append(_make_entry(response_text, "assistant"))
                    turn_count += 1
                    await _schedule_snapshot_if_due()
            finally:
                await _stop_snapshot_worker()


if __name__ == "__main__":
    asyncio.run(main())
