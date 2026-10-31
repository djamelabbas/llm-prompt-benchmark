# LLM & Prompt Optimization Lab

**A reproducible experimentation workbench for comparing LLMs, prompt versions, response quality, latency, token usage, and estimated inference costs.**

The project provides a Streamlit interface backed by Gemini, Langfuse observability, and a configurable LLM-as-a-Judge. It is designed to make prompt/model trade-offs measurable rather than subjective.

> **Status:** Portfolio / experimental project. The included insurance-related scenarios are fictional and do not represent a production insurance decision system.

## What it does

- **Compare models:** Run the same prompt and test cases against two Gemini models.
- **Compare prompts:** Evaluate multiple immutable prompt versions against the same model.
- **Manage prompt versions:** Create new versions from the Streamlit UI without overwriting previous versions.
- **Inspect each response:** Compare generated answers with references, evaluation criteria, token usage, latency, and trace identifiers.
- **Measure consistency:** Repeat experiments and aggregate results across runs.
- **Estimate costs:** Apply the rates in `pricing.json` to token usage, including reasoning tokens where reported.
- **Evaluate open-ended answers:** Use a structured Gemini LLM-as-a-Judge to assess factual accuracy, grounding, criteria satisfaction, and relevance.
- **Trace experiments:** Publish generation traces and evaluation scores to Langfuse.
- **Keep an audit trail:** Store experimental results and judge evaluations as local JSONL files.

## Evaluation design

| Dataset | Use case | Quality assessment |
| --- | --- | --- |
| `classification` | Insurance request classification (AUTO / MRH / MOTO) | Deterministic exact match |
| `agent_assurance` | Fictional insurance customer-support questions | Structured LLM-as-a-Judge |

For open-ended responses, the judge returns:

- `factual_accuracy`: factual consistency with the supplied context.
- `grounded`: absence of unsupported factual claims.
- `criteria_met`: one Boolean per evaluation criterion.
- `relevance`: focus on the question, without unnecessary content.
- `passed`: `factual_accuracy AND grounded AND all(criteria_met)`.

**Relevance is tracked separately**: an answer may satisfy the required business criteria while still containing unnecessary material.

Judge scores are model-generated assessments, **not objective ground truth**. Human review and larger datasets are advisable before drawing operational conclusions.

## Tech stack

Python 3.11 · [uv](https://docs.astral.sh/uv/) · [Streamlit](https://streamlit.io/) · [Google Gen AI SDK](https://googleapis.github.io/python-genai/) · [Gemini](https://ai.google.dev/) · [Langfuse](https://langfuse.com/) · Pydantic · Pandas / Plotly · Pytest

The currently configured experiments use `gemini-3.5-flash-lite` and `gemini-3.6-flash`; the LLM-as-a-Judge uses `gemini-3.6-flash`.

## Quick start

### 1. Prerequisites

- Python 3.11 and `uv`.
- A Google AI Studio Gemini API key for live experiments.
- A Langfuse project and API keys for tracing and scores.

### 2. Install dependencies

```powershell
uv sync --locked
```

### 3. Configure environment

If you do not already have a `.env` file, copy the example and fill in your own credentials:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Do not commit `.env`. The Langfuse endpoint in the example targets the **EU Cloud region**; change it if your Langfuse project belongs to another region.

### 4. Start the interface

```powershell
uv run streamlit run app.py
```

Select a dataset and a comparison mode, choose models or prompt versions, set the number of test cases, and confirm billable calls before starting a run.

### 5. Run tests (offline)

```powershell
uv run python -m pytest -q
```

The existing suite contains **22 automated tests**, covering prompt versioning, cost calculations, Gemini/Langfuse mocks, and the judge. Tests use synthetic fixtures or mocks, so they do not intentionally make billable Gemini calls.

## Command-line evaluation

Evaluate a previously generated `agent_assurance` benchmark file:

```powershell
uv run python judge.py --file results/benchmark_YOUR_RUN.jsonl --limit 2
```

The judge publishes its scores to the original Langfuse trace and writes a new `results/judge_*.jsonl` file. The CLI limits each invocation to **10 responses**.

Repeated experiments and result analysis are also supported by `repeat_experiment.py`, `analyze_experiment.py`, and `analyze_results.py`. Consult each script's arguments before running live benchmarks: API calls incur usage-based charges.

## Project structure

```text
.
├── app.py                   # Streamlit experiment dashboard
├── benchmark.py             # Gemini benchmark execution + tracing
├── prompt_store.py          # Prompt version management
├── judge.py                 # Structured LLM-as-a-Judge + Langfuse scores
├── calibrate_judge.py       # Judge calibration checks
├── repeat_experiment.py     # Multi-run experiment orchestration
├── analyze_results.py       # Benchmark analysis
├── analyze_experiment.py    # Repeated-experiment analysis
├── pricing.json             # Configurable inference pricing
├── datasets/                # Fictional evaluation cases (JSONL)
├── prompts/                 # Versioned prompt templates
├── tests/                   # Pytest suite
├── .github/workflows/       # Continuous integration
└── results/                 # Generated locally; excluded from Git
```

## How to interpret results

Compare configurations on more than one dimension: correctness or judge compliance, relevance, latency, token usage, estimated inference cost, and stability across repetitions. A shorter prompt does **not** automatically produce cheaper or better responses, especially when reasoning-token usage changes.

**Important limitations**

- Results from small test sets are illustrative and should not be generalized without further testing.
- LLM judge assessments can vary and may have biases; calibrate against human-labelled examples.
- Token usage and prices are provider-/model-dependent; verify and update `pricing.json` when rates change.
- Estimated benchmark inference cost does not necessarily include all downstream costs, such as separate judge calls or Langfuse hosting.
- API access and model availability depend on the configured provider account.

## Continuous integration

The GitHub Actions workflow runs the automated tests on pushes and pull requests against `main`. API credentials are **not required for the mocked test suite** and should never be added to repository files or CI logs.

## Roadmap

- Add support for additional providers through a LiteLLM-compatible gateway.
- Expand human-reviewed test datasets and judge calibration.
- Add experiment-level comparison exports and regression thresholds.

## Security and data

Only fictional demonstration scenarios should be committed. Keep real prompts, customer information, credentials, traces, and generated experiment results out of a public repository unless explicitly reviewed and sanitized.
