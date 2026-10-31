
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import judge


def sample_row():
    return {
        "run_id": "test-run-001",
        "dataset": "agent_assurance",
        "test_id": "agent-001",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "v1",
        "input": "Quel est le delai de declaration ?",
        "context": "Le delai est de 5 jours ouvres.",
        "reference": "5 jours ouvres.",
        "criteria": [
            "Mentionner 5 jours ouvres",
            "Ne pas inventer de delai",
        ],
        "answer": "Le delai est de 5 jours ouvres.",
        "error": None,
        "langfuse_trace_id": "trace-test",
        "langfuse_observation_id": "observation-test",
    }


@pytest.mark.parametrize(
    "factual,relevant,expected_passed",
    [
        (True, True, True),
        (True, False, True),
        (False, True, False),
    ],
)
def test_evaluate_verdict(
    factual, relevant, expected_passed
):
    client = MagicMock()

    verdict = judge.Judgment(
        factual_accuracy=factual,
        grounded=True,
        criteria_met=[True, True],
        relevance=relevant,
        reason="Evaluation simulee",
    )

    client.models.generate_content.return_value = (
        SimpleNamespace(
            parsed=verdict,
            text=None,
            model_version="gemini-3.6-flash",
            usage_metadata=SimpleNamespace(
                total_token_count=100
            ),
        )
    )

    result = judge.evaluate(
        client, sample_row()
    )

    assert result["passed"] is expected_passed
    assert result["relevance"] is relevant
    assert result["judge_tokens"] == 100

    client.models.generate_content.assert_called_once()


def test_wrong_number_of_criteria():
    client = MagicMock()

    client.models.generate_content.return_value = (
        SimpleNamespace(
            parsed=judge.Judgment(
                factual_accuracy=True,
                grounded=True,
                criteria_met=[True],
                relevance=True,
                reason="Test",
            ),
        )
    )

    with pytest.raises(ValueError, match="criteres"):
        judge.evaluate(client, sample_row())


def test_publish_langfuse_scores(monkeypatch):
    langfuse = MagicMock()

    monkeypatch.setattr(
        judge, "get_client", lambda: langfuse
    )

    evaluation = {
        "factual_accuracy": True,
        "grounded": True,
        "relevance": False,
        "passed": True,
        "judge_model": "gemini-3.6-flash",
        "reason": "Contenu superflu",
    }

    judge.publish_scores(sample_row(), evaluation)

    calls = langfuse.create_score.call_args_list

    assert len(calls) == 4

    scores = {
        call.kwargs["name"]: call.kwargs["value"]
        for call in calls
    }

    assert scores == {
        "judge_factual_accuracy": 1,
        "judge_grounded": 1,
        "judge_relevance": 0,
        "judge_passed": 1,
    }

    for call in calls:
        assert call.kwargs["trace_id"] == "trace-test"
        assert (
            call.kwargs["observation_id"]
            == "observation-test"
        )


def test_run_judge_without_external_calls(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "GEMINI_API_KEY", "fake-test-key"
    )

    (tmp_path / "results").mkdir()

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(sample_row()) + "\n",
        encoding="utf-8",
    )

    fake_client = MagicMock()
    fake_langfuse = MagicMock()

    monkeypatch.setattr(
        judge.genai,
        "Client",
        MagicMock(return_value=fake_client),
    )

    monkeypatch.setattr(
        judge,
        "get_client",
        lambda: fake_langfuse,
    )

    evaluation = {
        "passed": True,
        "relevance": True,
        "reason": "Test reussi",
        "factual_accuracy": True,
        "grounded": True,
        "criteria_met": [True, True],
        "judge_model": "gemini-3.6-flash",
        "judge_tokens": 100,
    }

    evaluate_mock = MagicMock(
        return_value=evaluation
    )
    publish_mock = MagicMock()

    monkeypatch.setattr(
        judge, "evaluate", evaluate_mock
    )
    monkeypatch.setattr(
        judge, "publish_scores", publish_mock
    )

    output = judge.run_judge(
        source_file=source,
        limit=1,
    )

    assert output.exists()

    saved = json.loads(
        output.read_text(encoding="utf-8").strip()
    )

    assert saved["evaluation"]["passed"] is True
    assert saved["error"] is None

    evaluate_mock.assert_called_once()
    publish_mock.assert_called_once()
    fake_client.close.assert_called_once()
    fake_langfuse.flush.assert_called_once()
