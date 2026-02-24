from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from .snapshot_store import load_snapshot_json, validate_snapshot_payload, VECTOR_INDEX_NAME


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
NEGATION_TOKENS = {"not", "no", "never", "false", "isnt", "isn't", "arent", "aren't"}
HALLUCINATION_SOURCES = {"llm", "model", "assistant"}
HALLUCINATION_CONFIDENCE_MAX = 0.5


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    content: str
    source: str
    confidence: float
    timestamp: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    timestamp: Optional[str]
    entries: List[MemoryEntry]


@dataclass(frozen=True)
class DriftCandidate:
    old_entry: Optional[MemoryEntry]
    new_entry: Optional[MemoryEntry]
    similarity: float
    drift_type: str
    reason: str


@dataclass(frozen=True)
class DiffResult:
    added: List[MemoryEntry]
    removed: List[MemoryEntry]
    changed: List[Tuple[MemoryEntry, MemoryEntry]]
    drifts: List[DriftCandidate]


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / len(sa | sb)


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _similarity(a: MemoryEntry, b: MemoryEntry) -> float:
    if a.embedding and b.embedding:
        return _cosine(a.embedding, b.embedding)
    return _jaccard(_tokens(a.content), _tokens(b.content))


def _contains_negation(tokens: Iterable[str]) -> bool:
    return any(t in NEGATION_TOKENS for t in tokens)


def _contradiction_heuristic(a: MemoryEntry, b: MemoryEntry) -> bool:
    tokens_a = _tokens(a.content)
    tokens_b = _tokens(b.content)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(set(tokens_a) & set(tokens_b))
    if overlap < 3:
        return False
    neg_a = _contains_negation(tokens_a)
    neg_b = _contains_negation(tokens_b)
    return neg_a != neg_b


def _extract_years(text: str) -> List[int]:
    years = []
    for match in re.findall(r"\b(19|20)\d{2}\b", text):
        years.append(int(match))
    return years


def _is_stale(entry: MemoryEntry, now: datetime) -> bool:
    if entry.timestamp:
        try:
            ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            if (now - ts).days > 365:
                return True
        except ValueError:
            pass
    years = _extract_years(entry.content)
    if years and max(years) < now.year - 1:
        return True
    return False


def _preference_key(entry: MemoryEntry) -> str:
    key = entry.metadata.get("preference_key")
    return str(key).strip().lower() if key else ""


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_hallucination(entry: MemoryEntry, has_match: bool, similarity: float) -> bool:
    if has_match:
        return False
    source = entry.source.lower().strip()
    provenance = str(entry.metadata.get("provenance", "")).lower().strip()
    if source in HALLUCINATION_SOURCES or provenance in HALLUCINATION_SOURCES:
        if entry.confidence < HALLUCINATION_CONFIDENCE_MAX and similarity < 0.3:
            return True
    return False


def _entry_from_json(obj: Dict[str, Any]) -> MemoryEntry:
    return MemoryEntry(
        entry_id=str(obj.get("id") or obj.get("entry_id") or ""),
        content=str(obj["content"]),
        source=str(obj.get("source", "")),
        confidence=float(obj.get("confidence", 0.0)),
        timestamp=obj.get("timestamp"),
        tags=list(obj.get("tags", [])),
        metadata=dict(obj.get("metadata", {})),
        embedding=list(obj.get("embedding")) if obj.get("embedding") else None,
    )


def load_snapshot(path: str | Path) -> Snapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_snapshot_payload(payload)
    entries = [_entry_from_json(e) for e in payload.get("entries", [])]
    return Snapshot(
        snapshot_id=str(payload.get("snapshot_id", "")),
        timestamp=payload.get("timestamp"),
        entries=entries,
    )


def diff_snapshots(old: Snapshot, new: Snapshot) -> DiffResult:
    added: List[MemoryEntry] = []
    removed: List[MemoryEntry] = []
    changed: List[Tuple[MemoryEntry, MemoryEntry]] = []
    drifts: List[DriftCandidate] = []

    used_old: set[int] = set()
    similarity_threshold = 0.7
    redundancy_threshold = 0.92
    old_pref_index: Dict[str, int] = {}
    for i, old_entry in enumerate(old.entries):
        key = _preference_key(old_entry)
        if key:
            old_pref_index[key] = i

    for new_entry in new.entries:
        best_idx = None
        best_sim = 0.0
        pref_key = _preference_key(new_entry)
        if pref_key and pref_key in old_pref_index:
            best_idx = old_pref_index[pref_key]
            best_sim = 1.0
        for i, old_entry in enumerate(old.entries):
            sim = _similarity(old_entry, new_entry)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx is None or best_sim < similarity_threshold:
            added.append(new_entry)
            if _is_hallucination(new_entry, has_match=False, similarity=best_sim):
                drift_type = "hallucination"
                reason = "Low-confidence LLM entry with no close match in old snapshot."
            else:
                drift_type = "added"
                reason = "No sufficiently similar entry in old snapshot."
            drifts.append(
                DriftCandidate(
                    old_entry=None,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type=drift_type,
                    reason=reason,
                )
            )
            continue

        used_old.add(best_idx)
        old_entry = old.entries[best_idx]
        if old_entry.content != new_entry.content:
            changed.append((old_entry, new_entry))

        now = datetime.now(timezone.utc)
        if _contradiction_heuristic(old_entry, new_entry):
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="contradiction",
                    reason="High overlap with opposing negation polarity.",
                )
            )
        elif _preference_key(old_entry) and _preference_key(old_entry) == _preference_key(new_entry):
            if old_entry.content != new_entry.content:
                drifts.append(
                    DriftCandidate(
                        old_entry=old_entry,
                        new_entry=new_entry,
                        similarity=best_sim,
                        drift_type="preference_update",
                        reason="Same preference key with updated value.",
                    )
                )
        elif _is_stale(new_entry, now):
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="staleness",
                    reason="Timestamp or year hints indicate staleness.",
                )
            )
        elif best_sim >= redundancy_threshold and old_entry.content == new_entry.content:
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="redundancy",
                    reason="Highly similar entries across snapshots.",
                )
            )

    for i, old_entry in enumerate(old.entries):
        if i not in used_old:
            removed.append(old_entry)
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=None,
                    similarity=0.0,
                    drift_type="removed",
                    reason="Entry not found in new snapshot.",
                )
            )

    return DiffResult(
        added=added,
        removed=removed,
        changed=changed,
        drifts=drifts,
    )


def _snapshot_stats(snapshot: Snapshot) -> Tuple[int, float]:
    return len(snapshot.entries), sum(e.confidence for e in snapshot.entries)


def _summarize_drifts(drifts: List[DriftCandidate], limit: int = 8) -> List[str]:
    lines: List[str] = []
    for drift in drifts[:limit]:
        if drift.new_entry and drift.old_entry:
            lines.append(
                f"- {drift.drift_type}: {drift.new_entry.content[:120]}"
            )
        elif drift.new_entry:
            lines.append(f"- {drift.drift_type}: {drift.new_entry.content[:120]}")
        elif drift.old_entry:
            lines.append(f"- {drift.drift_type}: {drift.old_entry.content[:120]}")
    if len(drifts) > limit:
        lines.append(f"...and {len(drifts) - limit} more drift items")
    return lines


def diff_snapshots_from_files(old_path: str, new_path: str) -> str:
    old = load_snapshot(old_path)
    new = load_snapshot(new_path)
    diff = diff_snapshots(old, new)

    old_count, old_conf = _snapshot_stats(old)
    new_count, new_conf = _snapshot_stats(new)

    lines = [
        f"Old snapshot: {old_count} entries, total confidence {old_conf:.2f}",
        f"New snapshot: {new_count} entries, total confidence {new_conf:.2f}",
        f"Added entries: {len(diff.added)}",
        f"Removed entries: {len(diff.removed)}",
        f"Changed entries: {len(diff.changed)}",
        f"Drift candidates: {len(diff.drifts)}",
    ]
    lines.extend(_summarize_drifts(diff.drifts))
    return "\n".join(lines)


def diff_snapshots_by_id(old_id: str, new_id: str, redis_url: str) -> str:
    old_payload = load_snapshot_json(old_id, redis_url)
    new_payload = load_snapshot_json(new_id, redis_url)

    validate_snapshot_payload(old_payload)
    validate_snapshot_payload(new_payload)

    old = Snapshot(
        snapshot_id=str(old_payload.get("snapshot_id", "")),
        timestamp=old_payload.get("timestamp"),
        entries=[_entry_from_json(e) for e in old_payload.get("entries", [])],
    )
    new = Snapshot(
        snapshot_id=str(new_payload.get("snapshot_id", "")),
        timestamp=new_payload.get("timestamp"),
        entries=[_entry_from_json(e) for e in new_payload.get("entries", [])],
    )

    use_vector = os.getenv("USE_REDIS_VECTOR_SEARCH", "false").strip().lower() in {"1", "true", "yes"}
    if use_vector:
        try:
            logging.info("Vector drift detection enabled for snapshots %s -> %s", old_id, new_id)
            diff = _diff_snapshots_with_vector(old, new, redis_url)
        except Exception:
            logging.exception("Vector drift detection failed; falling back to heuristic diff")
            diff = diff_snapshots(old, new)
    else:
        logging.info("Vector drift detection disabled; using heuristic diff")
        diff = diff_snapshots(old, new)

    old_count, old_conf = _snapshot_stats(old)
    new_count, new_conf = _snapshot_stats(new)

    lines = [
        f"Old snapshot: {old_count} entries, total confidence {old_conf:.2f}",
        f"New snapshot: {new_count} entries, total confidence {new_conf:.2f}",
        f"Added entries: {len(diff.added)}",
        f"Removed entries: {len(diff.removed)}",
        f"Changed entries: {len(diff.changed)}",
        f"Drift candidates: {len(diff.drifts)}",
    ]
    lines.extend(_summarize_drifts(diff.drifts))
    return "\n".join(lines)


def _vector_search(
    redis_url: str,
    snapshot_id: str,
    embedding: List[float],
    top_k: int = 1,
) -> List[Tuple[str, float]]:
    import struct
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    query = f"@snapshot_id:{{{snapshot_id}}}=>[KNN {top_k} @embedding $vec AS score]"
    params = {"vec": struct.pack(f"{len(embedding)}f", *embedding)}
    try:
        res = client.ft(VECTOR_INDEX_NAME).search(
            query,
            query_params=params,
            sort_by="score",
            return_fields=["entry_id", "score"],
        )
    except Exception:
        return []
    hits = []
    for doc in res.docs:
        entry_id = getattr(doc, "entry_id", "")
        score = float(getattr(doc, "score", 1.0))
        hits.append((entry_id, score))
    return hits


def _diff_snapshots_with_vector(old: Snapshot, new: Snapshot, redis_url: str) -> DiffResult:
    added: List[MemoryEntry] = []
    removed: List[MemoryEntry] = []
    changed: List[Tuple[MemoryEntry, MemoryEntry]] = []
    drifts: List[DriftCandidate] = []

    used_old: set[int] = set()
    similarity_threshold = 0.7
    redundancy_threshold = 0.92
    old_by_id = {e.entry_id: (i, e) for i, e in enumerate(old.entries)}

    for new_entry in new.entries:
        best_idx = None
        best_sim = 0.0
        if new_entry.embedding:
            hits = _vector_search(redis_url, old.snapshot_id, new_entry.embedding, top_k=1)
            if hits:
                entry_id, score = hits[0]
                if entry_id in old_by_id:
                    best_idx = old_by_id[entry_id][0]
                    best_sim = max(0.0, 1.0 - score)
        if best_idx is None:
            for i, old_entry in enumerate(old.entries):
                sim = _similarity(old_entry, new_entry)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
        if best_idx is None or best_sim < similarity_threshold:
            added.append(new_entry)
            if _is_hallucination(new_entry, has_match=False, similarity=best_sim):
                drift_type = "hallucination"
                reason = "Low-confidence LLM entry with no close match in old snapshot."
            else:
                drift_type = "added"
                reason = "No sufficiently similar entry in old snapshot."
            drifts.append(
                DriftCandidate(
                    old_entry=None,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type=drift_type,
                    reason=reason,
                )
            )
            continue

        used_old.add(best_idx)
        old_entry = old.entries[best_idx]
        if old_entry.content != new_entry.content:
            changed.append((old_entry, new_entry))

        now = datetime.now(timezone.utc)
        if _contradiction_heuristic(old_entry, new_entry):
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="contradiction",
                    reason="High overlap with opposing negation polarity.",
                )
            )
        elif _preference_key(old_entry) and _preference_key(old_entry) == _preference_key(new_entry):
            if old_entry.content != new_entry.content:
                drifts.append(
                    DriftCandidate(
                        old_entry=old_entry,
                        new_entry=new_entry,
                        similarity=best_sim,
                        drift_type="preference_update",
                        reason="Same preference key with updated value.",
                    )
                )
        elif _is_stale(new_entry, now):
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="staleness",
                    reason="Timestamp or year hints indicate staleness.",
                )
            )
        elif best_sim >= redundancy_threshold and old_entry.content == new_entry.content:
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=new_entry,
                    similarity=best_sim,
                    drift_type="redundancy",
                    reason="Highly similar entries across snapshots.",
                )
            )

    for i, old_entry in enumerate(old.entries):
        if i not in used_old:
            removed.append(old_entry)
            drifts.append(
                DriftCandidate(
                    old_entry=old_entry,
                    new_entry=None,
                    similarity=0.0,
                    drift_type="removed",
                    reason="Entry not found in new snapshot.",
                )
            )

    return DiffResult(
        added=added,
        removed=removed,
        changed=changed,
        drifts=drifts,
    )


def _build_rag_context(
    target: MemoryEntry, pool: List[MemoryEntry], top_k: int = 5
) -> List[MemoryEntry]:
    scored = []
    for entry in pool:
        if entry.entry_id == target.entry_id:
            continue
        scored.append((_similarity(target, entry), entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


async def repair_snapshot_with_rewriter(
    old_id: str,
    new_id: str,
    repaired_id: str,
    redis_url: str,
    rewrite_fn: Callable[[MemoryEntry, List[MemoryEntry], DriftCandidate], Awaitable[Optional[str]]],
    reflect_fn: Optional[
        Callable[[MemoryEntry, List[MemoryEntry], DriftCandidate], Awaitable[Optional[Dict[str, Any]]]]
    ] = None,
) -> Dict[str, Any]:
    old_payload = load_snapshot_json(old_id, redis_url)
    new_payload = load_snapshot_json(new_id, redis_url)

    validate_snapshot_payload(old_payload)
    validate_snapshot_payload(new_payload)

    old = Snapshot(
        snapshot_id=str(old_payload.get("snapshot_id", "")),
        timestamp=old_payload.get("timestamp"),
        entries=[_entry_from_json(e) for e in old_payload.get("entries", [])],
    )
    new = Snapshot(
        snapshot_id=str(new_payload.get("snapshot_id", "")),
        timestamp=new_payload.get("timestamp"),
        entries=[_entry_from_json(e) for e in new_payload.get("entries", [])],
    )

    diff = diff_snapshots(old, new)
    pool = old.entries + new.entries

    repaired_entries: List[Dict[str, Any]] = []
    deleted_entries: List[str] = []
    latest_pref: Dict[str, str] = {}
    latest_pref_ts: Dict[str, datetime] = {}
    for entry in new.entries:
        key = _preference_key(entry)
        if not key:
            continue
        ts = _parse_ts(entry.timestamp) or datetime.min.replace(tzinfo=timezone.utc)
        if key not in latest_pref_ts or ts > latest_pref_ts[key]:
            latest_pref[key] = entry.entry_id
            latest_pref_ts[key] = ts

    for entry in new.entries:
        pref_key = _preference_key(entry)
        if pref_key and latest_pref.get(pref_key) != entry.entry_id:
            deleted_entries.append(entry.entry_id)
            continue
        entry_dict = {
            "id": entry.entry_id,
            "content": entry.content,
            "source": entry.source,
            "confidence": entry.confidence,
            "tags": entry.tags,
            "metadata": dict(entry.metadata),
        }
        if entry.timestamp is not None:
            entry_dict["timestamp"] = entry.timestamp
        if entry.embedding is not None:
            entry_dict["embedding"] = entry.embedding

        delete_entry = False
        for drift in diff.drifts:
            if drift.new_entry and drift.new_entry.entry_id == entry.entry_id:
                entry_dict["metadata"]["drift_type"] = drift.drift_type
                entry_dict["metadata"]["drift_reason"] = drift.reason
                rag_context = _build_rag_context(entry, pool)
                decision = None
                if reflect_fn:
                    decision = await reflect_fn(entry, rag_context, drift)
                    if decision:
                        entry_dict["metadata"]["reflective_assessment"] = decision.get("assessment")
                        entry_dict["metadata"]["reflective_action"] = decision.get("action")
                        entry_dict["metadata"]["reflective_confidence"] = decision.get("confidence")
                        entry_dict["metadata"]["reflective_rationale"] = decision.get("rationale")

                action = None
                if decision:
                    action = str(decision.get("action", "")).lower().strip()

                if action in {"delete", "remove"}:
                    entry_dict["metadata"]["repair_action"] = "delete"
                    deleted_entries.append(entry.entry_id)
                    delete_entry = True
                elif action in {"rewrite", "update"}:
                    rewritten = await rewrite_fn(entry, rag_context, drift)
                    if rewritten:
                        entry_dict["content"] = rewritten
                        entry_dict["metadata"]["repair_action"] = "rewrite"
                        entry_dict["confidence"] = max(entry.confidence, 0.75)
                    else:
                        entry_dict["metadata"]["repair_action"] = "annotate"
                elif action in {"annotate", "flag"}:
                    entry_dict["metadata"]["repair_action"] = "annotate"
                elif action in {"keep", "keep_new"}:
                    entry_dict["metadata"]["repair_action"] = "keep_new"
                elif drift.drift_type == "preference_update":
                    entry_dict["metadata"]["repair_action"] = "keep_new"
                elif drift.drift_type in {"contradiction", "staleness"}:
                    rewritten = await rewrite_fn(entry, rag_context, drift)
                    if rewritten:
                        entry_dict["content"] = rewritten
                        entry_dict["metadata"]["repair_action"] = "rewrite"
                        entry_dict["confidence"] = max(entry.confidence, 0.75)
                    else:
                        entry_dict["metadata"]["repair_action"] = "annotate"
                elif drift.drift_type == "hallucination":
                    rewritten = await rewrite_fn(entry, rag_context, drift)
                    if rewritten:
                        entry_dict["content"] = rewritten
                        entry_dict["metadata"]["repair_action"] = "rewrite"
                        entry_dict["confidence"] = max(entry.confidence, 0.75)
                    else:
                        entry_dict["metadata"]["repair_action"] = "delete"
                        deleted_entries.append(entry.entry_id)
                        delete_entry = True
                elif drift.drift_type == "redundancy":
                    entry_dict["metadata"]["repair_action"] = "annotate"
                else:
                    entry_dict["metadata"]["repair_action"] = "keep_new"
                break

        if not delete_entry:
            repaired_entries.append(entry_dict)

    # Carry forward old entries that are not similar to any new entry
    for old_entry in old.entries:
        pref_key = _preference_key(old_entry)
        if pref_key and pref_key in latest_pref:
            continue
        if all(_similarity(old_entry, n) < 0.7 for n in new.entries):
            carried = {
                "id": old_entry.entry_id,
                "content": old_entry.content,
                "source": old_entry.source,
                "confidence": old_entry.confidence,
                "tags": old_entry.tags,
                "metadata": dict(old_entry.metadata),
            }
            if old_entry.timestamp is not None:
                carried["timestamp"] = old_entry.timestamp
            if old_entry.embedding is not None:
                carried["embedding"] = old_entry.embedding
            repaired_entries.append(carried)

    repaired_payload = {
        "snapshot_id": repaired_id,
        "entries": repaired_entries,
        "metadata": {
            "repaired_from": {"old": old_id, "new": new_id},
            "drift_count": len(diff.drifts),
        },
    }
    if new_payload.get("timestamp") is not None:
        repaired_payload["timestamp"] = new_payload.get("timestamp")
    if deleted_entries:
        repaired_payload["metadata"]["deleted_entries"] = deleted_entries

    validate_snapshot_payload(repaired_payload)
    return repaired_payload


def repair_snapshot_by_id(
    old_id: str, new_id: str, repaired_id: str, redis_url: str
) -> Dict[str, Any]:
    async def _noop_rewriter(
        entry: MemoryEntry, context: List[MemoryEntry], drift: DriftCandidate
    ) -> Optional[str]:
        _ = (entry, context, drift)
        return None

    async def _noop_reflect(
        entry: MemoryEntry, context: List[MemoryEntry], drift: DriftCandidate
    ) -> Optional[Dict[str, Any]]:
        _ = (entry, context, drift)
        return None

    import asyncio

    return asyncio.run(
        repair_snapshot_with_rewriter(
            old_id,
            new_id,
            repaired_id,
            redis_url,
            _noop_rewriter,
            _noop_reflect,
        )
    )
