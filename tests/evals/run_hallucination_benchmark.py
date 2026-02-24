from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import reflective_memory_diffing_agent.memory_diff as memory_diff


@dataclass
class CaseResult:
    case_id: str
    hallucinations_before: int
    hallucinations_after: int


def _snapshot_from_payload(payload: Dict[str, Any]) -> memory_diff.Snapshot:
    entries = [memory_diff._entry_from_json(e) for e in payload.get("entries", [])]
    return memory_diff.Snapshot(
        snapshot_id=str(payload.get("snapshot_id", "")),
        timestamp=payload.get("timestamp"),
        entries=entries,
    )


def _count_hallucinations(old_payload: Dict[str, Any], new_payload: Dict[str, Any]) -> int:
    old_snapshot = _snapshot_from_payload(old_payload)
    new_snapshot = _snapshot_from_payload(new_payload)
    diff = memory_diff.diff_snapshots(old_snapshot, new_snapshot)
    return sum(1 for d in diff.drifts if d.drift_type == "hallucination")


def _repair_payload(old_payload: Dict[str, Any], new_payload: Dict[str, Any]) -> Dict[str, Any]:
    original_loader = memory_diff.load_snapshot_json
    try:
        memory_diff.load_snapshot_json = lambda sid, url: old_payload if sid == "OLD" else new_payload
        return memory_diff.repair_snapshot_by_id("OLD", "NEW", "REPAIRED", "redis://unused")
    finally:
        memory_diff.load_snapshot_json = original_loader


def run_benchmark(cases_path: Path) -> List[CaseResult]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    results: List[CaseResult] = []
    for case in payload.get("cases", []):
        old_payload = case["old"]
        new_payload = case["new"]

        before = _count_hallucinations(old_payload, new_payload)
        repaired_payload = _repair_payload(old_payload, new_payload)
        after = _count_hallucinations(old_payload, repaired_payload)

        results.append(
            CaseResult(
                case_id=case.get("id", "unknown"),
                hallucinations_before=before,
                hallucinations_after=after,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline hallucination-reduction benchmark.")
    parser.add_argument(
        "--cases",
        default="tests/evals/hallucination_cases.json",
        help="Path to benchmark cases json file.",
    )
    args = parser.parse_args()

    results = run_benchmark(Path(args.cases))
    total_before = sum(r.hallucinations_before for r in results)
    total_after = sum(r.hallucinations_after for r in results)
    reduction = 0.0
    if total_before > 0:
        reduction = (total_before - total_after) / total_before

    print("Hallucination Benchmark")
    print("=======================")
    for r in results:
        print(
            f"- {r.case_id}: before={r.hallucinations_before}, "
            f"after={r.hallucinations_after}"
        )
    print("-----------------------")
    print(f"Total before: {total_before}")
    print(f"Total after : {total_after}")
    print(f"Reduction   : {reduction:.2%}")


if __name__ == "__main__":
    main()
