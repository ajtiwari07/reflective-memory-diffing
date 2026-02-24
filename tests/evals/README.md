# Hallucination Benchmark

This folder contains an offline benchmark to validate hallucination reduction
from drift repair.

## What it measures

- `hallucinations_before`: hallucination drift count from `old -> new` snapshot
- `hallucinations_after`: hallucination drift count from `old -> repaired` snapshot
- overall reduction percentage across all cases

## Run

From repo root:

```powershell
$env:PYTHONPATH="src"
python tests/evals/run_hallucination_benchmark.py
```

Custom cases file:

```powershell
$env:PYTHONPATH="src"
python tests/evals/run_hallucination_benchmark.py --cases tests/evals/hallucination_cases.json
```

## Extend

Add new scenarios to `tests/evals/hallucination_cases.json` with:

- `id`
- `old` snapshot payload
- `new` snapshot payload

The benchmark will include each case in totals automatically.

## DeepEval (answer-level)

This project also includes a DeepEval harness for answer-level hallucination
scoring of model outputs.

Run:

```powershell
$env:PYTHONPATH="src"
python tests/evals/run_deepeval_hallucination.py --cases tests/evals/deepeval_cases.json --threshold 0.5
```

Notes:

- DeepEval requires an evaluation model.
- The runner auto-loads `config/.env` and supports:
  - OpenAI via `OPENAI_API_KEY`
  - Azure OpenAI via `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- Optional override: `DEEPEVAL_PROVIDER=azure` to force Azure provider path.
- Use this for A/B comparison of response quality, while the offline benchmark
  focuses on memory-level drift/hallucination counts.
