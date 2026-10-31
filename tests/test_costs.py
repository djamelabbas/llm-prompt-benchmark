
import pytest

from analyze_experiment import calculate_cost


TARIFFS = {
    "gemini-3.5-flash-lite": {
        "input": 0.30,
        "output": 2.50,
    }
}


def make_record(**overrides):
    record = {
        "run_id": "20261031_120000_000000",
        "model": "gemini-3.5-flash-lite",
        "model_version": "gemini-3.5-flash-lite",
        "input_tokens": 100,
        "output_tokens": 20,
        "thinking_tokens": 30,
        "total_tokens": 150,
    }

    record.update(overrides)
    return record


def test_cost_includes_thinking_tokens():
    record = make_record()

    cost = calculate_cost(record, TARIFFS)

    expected = (
        100 * 0.30 + (20 + 30) * 2.50
    ) / 1_000_000

    assert cost == pytest.approx(expected)


def test_cost_without_thinking_tokens():
    record = make_record(
        thinking_tokens=None,
        total_tokens=120,
    )

    cost = calculate_cost(record, TARIFFS)

    expected = (
        100 * 0.30 + 20 * 2.50
    ) / 1_000_000

    assert cost == pytest.approx(expected)


def test_inconsistent_token_count():
    record = make_record(total_tokens=999)

    with pytest.raises(
        ValueError, match="Comptage"
    ):
        calculate_cost(record, TARIFFS)


def test_unknown_model():
    record = make_record(
        model="unknown-model",
        model_version="unknown-model",
    )

    with pytest.raises(
        ValueError, match="Tarif manquant"
    ):
        calculate_cost(record, TARIFFS)


def test_expired_pricing():
    tariffs = {
        "gemini-3.5-flash-lite": {
            "input": 0.30,
            "output": 2.50,
            "valid_until": "2026-10-30",
        }
    }

    with pytest.raises(
        ValueError, match="Tarif expire"
    ):
        calculate_cost(make_record(), tariffs)


def test_missing_token_metrics():
    record = make_record(input_tokens=None)

    with pytest.raises(
        ValueError, match="Metriques"
    ):
        calculate_cost(record, TARIFFS)
