from reflective_memory_diffing_agent.memory_diff import MemoryEntry, Snapshot, diff_snapshots, repair_snapshot_by_id


class _FakeStore:
    def __init__(self, payloads):
        self.payloads = payloads

    def load(self, snapshot_id, redis_url):
        return self.payloads[snapshot_id]


def test_repair_snapshot_by_id(monkeypatch):
    old_payload = {
        "snapshot_id": "M_t-1",
        "entries": [
            {
                "id": "e1",
                "content": "The sky is blue.",
                "source": "external",
                "confidence": 0.9,
            }
        ],
    }
    new_payload = {
        "snapshot_id": "M_t",
        "entries": [
            {
                "id": "e2",
                "content": "The sky is not blue.",
                "source": "llm",
                "confidence": 0.4,
            }
        ],
    }

    monkeypatch.setattr(
        "reflective_memory_diffing_agent.memory_diff.load_snapshot_json",
        lambda sid, url: old_payload if sid == "M_t-1" else new_payload,
    )
    repaired = repair_snapshot_by_id("M_t-1", "M_t", "M_t_repaired", "redis://localhost:6379")

    assert repaired["snapshot_id"] == "M_t_repaired"
    assert len(repaired["entries"]) >= 1


def test_diff_snapshots_added_removed_changed():
    old = Snapshot(
        snapshot_id="M_t-1",
        timestamp=None,
        entries=[
            MemoryEntry(
                entry_id="e1",
                content="The capital of Peru is Lima.",
                source="external",
                confidence=0.9,
                timestamp="2026-01-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            )
        ],
    )
    new = Snapshot(
        snapshot_id="M_t",
        timestamp=None,
        entries=[
            MemoryEntry(
                entry_id="e1",
                content="The capital of Peru is Lima.",
                source="external",
                confidence=0.9,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
            MemoryEntry(
                entry_id="e2",
                content="The capital of Chile is Santiago.",
                source="external",
                confidence=0.8,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
        ],
    )

    diff = diff_snapshots(old, new)
    assert len(diff.added) == 1
    assert len(diff.removed) == 0
    assert len(diff.changed) == 0


def test_diff_snapshots_hallucination_detection():
    old = Snapshot(
        snapshot_id="M_t-1",
        timestamp=None,
        entries=[],
    )
    new = Snapshot(
        snapshot_id="M_t",
        timestamp=None,
        entries=[
            MemoryEntry(
                entry_id="e1",
                content="The moon is made of cheese.",
                source="llm",
                confidence=0.2,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            )
        ],
    )

    diff = diff_snapshots(old, new)
    assert len(diff.drifts) == 1
    assert diff.drifts[0].drift_type == "hallucination"


def test_diff_snapshots_multiple_drifts():
    old = Snapshot(
        snapshot_id="M_t-1",
        timestamp="2024-01-01T00:00:00Z",
        entries=[
            MemoryEntry(
                entry_id="e1",
                content="The capital of Peru is Lima.",
                source="external",
                confidence=0.9,
                timestamp="2024-01-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
            MemoryEntry(
                entry_id="e2",
                content="The sky is blue.",
                source="external",
                confidence=0.9,
                timestamp="2024-01-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
        ],
    )
    new = Snapshot(
        snapshot_id="M_t",
        timestamp="2026-02-01T00:00:00Z",
        entries=[
            MemoryEntry(
                entry_id="e1",
                content="The capital of Peru is Lima.",
                source="external",
                confidence=0.9,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
            MemoryEntry(
                entry_id="e3",
                content="The sky is not blue.",
                source="llm",
                confidence=0.4,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
            MemoryEntry(
                entry_id="e4",
                content="XQZ blorf glanx.",
                source="llm",
                confidence=0.2,
                timestamp="2026-02-01T00:00:00Z",
                tags=[],
                metadata={},
                embedding=None,
            ),
        ],
    )

    diff = diff_snapshots(old, new)
    drift_types = {d.drift_type for d in diff.drifts}
    assert "contradiction" in drift_types
    assert "hallucination" in drift_types


def test_repair_snapshot_deletes_hallucination(monkeypatch):
    old_payload = {
        "snapshot_id": "M_t-1",
        "entries": [],
    }
    new_payload = {
        "snapshot_id": "M_t",
        "entries": [
            {
                "id": "e1",
                "content": "XQZ blorf glanx.",
                "source": "llm",
                "confidence": 0.2,
                "timestamp": "2026-02-01T00:00:00Z",
            }
        ],
    }

    monkeypatch.setattr(
        "reflective_memory_diffing_agent.memory_diff.load_snapshot_json",
        lambda sid, url: old_payload if sid == "M_t-1" else new_payload,
    )
    repaired = repair_snapshot_by_id("M_t-1", "M_t", "M_t_repaired", "redis://localhost:6379")

    assert repaired["snapshot_id"] == "M_t_repaired"
    assert repaired["entries"] == []
    assert repaired["metadata"]["drift_count"] == 1
    assert repaired["metadata"]["deleted_entries"] == ["e1"]


def test_repair_snapshot_multiple_drifts(monkeypatch):
    old_payload = {
        "snapshot_id": "M_t-1",
        "entries": [
            {
                "id": "e1",
                "content": "The sky is blue.",
                "source": "external",
                "confidence": 0.9,
            },
            {
                "id": "e2",
                "content": "The capital of Peru is Lima.",
                "source": "external",
                "confidence": 0.9,
            },
        ],
    }
    new_payload = {
        "snapshot_id": "M_t",
        "entries": [
            {
                "id": "e3",
                "content": "The sky is not blue.",
                "source": "llm",
                "confidence": 0.4,
            },
            {
                "id": "e4",
                "content": "XQZ blorf glanx.",
                "source": "llm",
                "confidence": 0.2,
            },
            {
                "id": "e2",
                "content": "The capital of Peru is Lima.",
                "source": "external",
                "confidence": 0.9,
            },
        ],
    }

    monkeypatch.setattr(
        "reflective_memory_diffing_agent.memory_diff.load_snapshot_json",
        lambda sid, url: old_payload if sid == "M_t-1" else new_payload,
    )
    repaired = repair_snapshot_by_id("M_t-1", "M_t", "M_t_repaired", "redis://localhost:6379")

    entries_by_id = {e["id"]: e for e in repaired["entries"]}
    assert "e4" not in entries_by_id
    assert repaired["metadata"]["deleted_entries"] == ["e4"]
    assert entries_by_id["e3"]["metadata"]["drift_type"] == "contradiction"
