from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from deepeval.metrics import HallucinationMetric
from deepeval.models import AzureOpenAIModel, GPTModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv


@dataclass
class EvalResult:
    case_id: str
    score: float
    passed: bool
    reason: str


def load_cases(path: Path) -> List[LLMTestCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[LLMTestCase] = []
    for case in payload.get("cases", []):
        out.append(
            LLMTestCase(
                input=case["input"],
                actual_output=case["actual_output"],
                expected_output=case.get("expected_output", ""),
                retrieval_context=case.get("context", []),
                context=case.get("context", []),
            )
        )
    return out


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_env = repo_root / "config" / ".env"
    if config_env.exists():
        load_dotenv(dotenv_path=config_env)
    load_dotenv()


def _build_eval_model():
    provider = os.getenv("DEEPEVAL_PROVIDER", "").strip().lower()
    if provider in {"azure", "azure_openai"}:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
        if not endpoint or not api_key or not deployment:
            raise RuntimeError(
                "Azure DeepEval requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
            )
        return AzureOpenAIModel(
            model=deployment,
            deployment_name=deployment,
            api_key=api_key,
            base_url=endpoint,
            api_version=api_version,
        )

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        model_name = os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini").strip()
        return GPTModel(model=model_name, api_key=openai_key)

    # Auto-fallback to Azure config if OpenAI key is not present.
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
    if endpoint and api_key and deployment:
        return AzureOpenAIModel(
            model=deployment,
            deployment_name=deployment,
            api_key=api_key,
            base_url=endpoint,
            api_version=api_version,
        )

    raise RuntimeError(
        "No LLM credentials found for DeepEval. Set OPENAI_API_KEY, or set "
        "AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY/AZURE_OPENAI_DEPLOYMENT."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval hallucination metric on answer samples.")
    parser.add_argument(
        "--cases",
        default="tests/evals/deepeval_cases.json",
        help="Path to deepeval cases file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hallucination threshold (lower is stricter).",
    )
    args = parser.parse_args()

    _load_env()
    cases = load_cases(Path(args.cases))
    model = _build_eval_model()
    metric = HallucinationMetric(threshold=args.threshold, model=model)

    results: List[EvalResult] = []
    for i, case in enumerate(cases, start=1):
        metric.measure(case)
        results.append(
            EvalResult(
                case_id=f"case_{i}",
                score=float(metric.score),
                passed=bool(metric.success),
                reason=str(metric.reason),
            )
        )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0.0

    print("DeepEval Hallucination Results")
    print("==============================")
    for r in results:
        print(f"- {r.case_id}: score={r.score:.4f}, pass={r.passed}, reason={r.reason}")
    print("------------------------------")
    print(f"Passed: {passed}/{total}")
    print(f"Average score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
