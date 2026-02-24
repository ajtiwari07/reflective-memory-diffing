import pytest

from reflective_memory_diffing_agent.snapshot_store import validate_snapshot_payload


def test_validate_snapshot_payload_success():
    payload = {
        "snapshot_id": "M_t",
        "timestamp": "2026-02-12T00:00:00Z",
        "entries": [
            {
                "id": "e1",
                "content": "The capital of Peru is Lima.",
                "source": "external",
                "confidence": 0.9,
            }
        ],
    }
    validate_snapshot_payload(payload)


def test_validate_snapshot_payload_missing_content():
    payload = {
        "snapshot_id": "M_t",
        "entries": [
            {
                "id": "e1",
                "source": "external",
            }
        ],
    }
    with pytest.raises(ValueError):
        validate_snapshot_payload(payload)
