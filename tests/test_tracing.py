
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import benchmark


def make_record(dataset="classification", expected="AUTO"):
    return {
        "run_id": "test-run-001",
        "mode": "prompts",
        "dataset": dataset,
        "test_id": "test-001",
        "prompt_version": "v1",
        "prompt_hash": "abc123",
        "expected": expected,
    }


@pytest.fixture
def mocked_services(monkeypatch):
    # Simuler Langfuse
    langfuse = MagicMock()
    observation = MagicMock()

    observation.trace_id = "trace-test"
    observation.id = "observation-test"

    langfuse.start_as_current_observation.return_value.__enter__.return_value = observation

    monkeypatch.setattr(
        benchmark,
        "get_client",
        lambda: langfuse,
    )

    # Simuler Gemini
    client = MagicMock()

    client.models.generate_content.return_value = SimpleNamespace(
        text="AUTO",
        model_version="gemini-3.5-flash-lite",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=20,
            total_token_count=130,
        ),
    )

    return client, langfuse, observation


def test_generation_and_token_tracking(mocked_services):
    client, langfuse, observation = mocked_services
    record = make_record()

    response = benchmark.generate_traced(
        client, "gemini-3.5-flash-lite", "Question test", record
    )

    assert response.text == "AUTO"
    assert record["langfuse_trace_id"] == "trace-test"
    assert record["langfuse_observation_id"] == "observation-test"

    usage = observation.update.call_args.kwargs["usage_details"]

    assert usage["input"] == 100
    assert usage["output"] == 10
    assert usage["output_reasoning"] == 20
    assert usage["total"] == 130

    client.models.generate_content.assert_called_once()


def test_correct_classification_scores_one(mocked_services):
    client, _, observation = mocked_services

    benchmark.generate_traced(
        client,
        "gemini-3.5-flash-lite",
        "Question test",
        make_record(expected="AUTO"),
    )

    assert observation.score.call_args.kwargs["value"] == 1


def test_wrong_classification_scores_zero(mocked_services):
    client, _, observation = mocked_services

    benchmark.generate_traced(
        client,
        "gemini-3.5-flash-lite",
        "Question test",
        make_record(expected="MRH"),
    )

    assert observation.score.call_args.kwargs["value"] == 0


def test_agent_answer_has_no_exact_match(mocked_services):
    client, _, observation = mocked_services

    benchmark.generate_traced(
        client,
        "gemini-3.5-flash-lite",
        "Question ouverte",
        make_record(dataset="agent_assurance", expected=None),
    )

    observation.score.assert_not_called()


def test_api_error_is_propagated(mocked_services):
    client, _, observation = mocked_services

    client.models.generate_content.side_effect = RuntimeError(
        "API Gemini indisponible"
    )

    with pytest.raises(RuntimeError, match="Gemini"):
        benchmark.generate_traced(
            client,
            "gemini-3.5-flash-lite",
            "Question test",
            make_record(),
        )

    observation.score.assert_not_called()


def test_run_benchmark_flushes_langfuse(monkeypatch):
    output = Path("results/test.jsonl")

    core = Mock(return_value=output)
    langfuse = MagicMock()

    monkeypatch.setattr(benchmark, "_run_benchmark", core)
    monkeypatch.setattr(
        benchmark, "get_client", lambda: langfuse
    )

    result = benchmark.run_benchmark(
        mode="prompts",
        dataset="agent_assurance",
        model="gemini-3.5-flash-lite",
        versions=["v1", "v3"],
        limit=1,
    )

    assert result == output
    assert core.call_args.kwargs["versions"] == ["v1", "v3"]
    langfuse.flush.assert_called_once()
