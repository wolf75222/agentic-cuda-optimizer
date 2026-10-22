"""Recorded usage attribution only; no study database, model or GPU."""
from copy import deepcopy
import json

import pytest

from optimizer_agent.suite_orchestrator import observed_metrics, write_json


ABSENT = object()


def model(*, call_id="call", purpose="research", elapsed=2, at=1, usage=None):
    payload = {"call_id": call_id, "reservation_id": "reservation-" + call_id,
               "usage": {"input_tokens": 3, "output_tokens": 2} if usage is None else usage}
    if purpose is not ABSENT:
        payload["purpose"] = purpose
    if elapsed is not ABSENT:
        payload["elapsed_seconds"] = elapsed
    return {"kind": "model", "created_at": at, "payload": payload}


def observe(tmp_path, events):
    path = tmp_path / "generation/events.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # External/legacy observations can contain nonfinite JSON extensions.
    # Keep the production writer strict and exercise the reader explicitly.
    path.write_text(json.dumps(events, allow_nan=True))
    write_json(tmp_path / "generation/clock.json", {"started_at": 0})
    before = path.read_bytes()
    metrics = observed_metrics(tmp_path, {})
    assert path.read_bytes() == before
    return metrics


@pytest.mark.parametrize("elapsed", [ABSENT, None, True, False, -1, -0.5,
                                    float("nan"), float("inf"), float("-inf"), "0", []])
def test_absent_or_invalid_duration_is_unknown_without_erasing_known_tokens(tmp_path, elapsed):
    metrics = observe(tmp_path, [model(elapsed=elapsed)])
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] is None
    assert metrics["research_model_wall_seconds_unknown_observations"] == 1
    assert metrics["research_model_tokens"] == 5
    assert "research_model_wall_seconds" in metrics["unknown_metrics"]


@pytest.mark.parametrize("elapsed", [0, 0.0, 1.25])
def test_recorded_nonnegative_finite_duration_including_zero_remains_known(tmp_path, elapsed):
    metrics = observe(tmp_path, [model(elapsed=elapsed)])
    assert metrics["research_model_wall_seconds"] == elapsed
    assert metrics["research_model_wall_seconds_known_lower_bound"] == elapsed
    assert metrics["research_model_wall_seconds_unknown_observations"] == 0
    assert "research_model_wall_seconds" not in metrics["unknown_metrics"]


@pytest.mark.parametrize("known", [0, 1.25])
def test_partial_duration_has_a_separate_known_lower_bound(tmp_path, known):
    metrics = observe(tmp_path, [model(call_id="known", elapsed=known),
                                model(call_id="unknown", elapsed=None)])
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] == known
    assert metrics["research_model_wall_seconds_unknown_observations"] == 1
    assert metrics["research_model_tokens"] == 10


@pytest.mark.parametrize("purpose", [ABSENT, None, "", "  ", False, {"research": "untyped"}])
def test_unknown_phase_does_not_establish_zero_or_assign_known_costs_to_research(tmp_path, purpose):
    metrics = observe(tmp_path, [model(purpose=purpose, elapsed=9)])
    assert metrics["research_model_tokens"] is None
    assert metrics["research_model_usage"]["known_model_tokens"] is None
    assert metrics["research_model_usage"]["unreported_calls"] == 1
    assert metrics["research_model_usage"]["observed_calls"] == 1
    assert metrics["model_observations_with_unknown_purpose"] == 1
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] is None


def test_unknown_phase_preserves_only_the_definitely_research_subtotals(tmp_path):
    metrics = observe(tmp_path, [model(call_id="research", elapsed=2),
        model(call_id="unknown", purpose=None, elapsed=9,
              usage={"input_tokens": 30, "output_tokens": 10})])
    assert metrics["research_model_tokens"] is None
    assert metrics["research_model_usage"]["known_model_tokens"] == 5
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] == 2
    assert metrics["research_model_wall_seconds_unknown_observations"] == 1


def test_matched_uncertainty_for_unknown_phase_does_not_count_a_second_call(tmp_path):
    metrics = observe(tmp_path, [model(purpose=None),
        {"kind": "model_uncertain", "created_at": 2, "payload": {
            "call_id": "call", "reservation_id": "reservation-call", "purpose": None}}])
    assert metrics["unmatched_model_uncertain_observations"] == 0
    assert metrics["research_model_usage"]["observed_calls"] == 1
    assert metrics["research_model_usage"]["unreported_calls"] == 1
    assert metrics["research_model_wall_seconds_unknown_observations"] == 1


def test_known_nonresearch_scope_can_be_zero_but_absent_evidence_cannot(tmp_path):
    known = observe(tmp_path, [model(purpose="proposal", elapsed=None)])
    assert known["research_model_tokens"] == 0
    assert known["research_model_wall_seconds"] == 0
    missing = observe(tmp_path, [])
    assert missing["research_model_tokens"] is None
    assert missing["research_model_wall_seconds"] is None
    assert missing["research_model_wall_seconds_known_lower_bound"] is None


@pytest.mark.parametrize("later", [model(at=3),
    {"kind": "model_uncertain", "created_at": 3, "payload": {
        "call_id": "later", "purpose": "research"}}])
def test_later_model_evidence_cannot_fabricate_zero_cost_before_first_gain(tmp_path, later):
    original = deepcopy(later)
    metrics = observe(tmp_path, [{"kind": "promotion", "created_at": 2}, later])
    assert metrics["model_tokens_to_first_useful_gain"] is None
    assert metrics["model_calls_to_first_useful_gain"] is None
    assert metrics["known_model_calls_to_first_useful_gain"] is None
    assert metrics["model_usage_to_first_useful_gain"] is None
    assert later == original


def test_first_gain_at_zero_time_keeps_a_proven_zero_token_call(tmp_path):
    metrics = observe(tmp_path, [model(at=0, elapsed=0,
        usage={"input_tokens": 0, "output_tokens": 0}),
        {"kind": "promotion", "created_at": 0}])
    assert metrics["model_tokens_to_first_useful_gain"] == 0
    assert metrics["model_calls_to_first_useful_gain"] == 1
    assert metrics["time_to_first_useful_gain_seconds"] == 0


def test_unmatched_research_uncertainty_does_not_erase_known_duration_subtotal(tmp_path):
    metrics = observe(tmp_path, [model(elapsed=2),
        {"kind": "model_uncertain", "created_at": 2, "payload": {
            "call_id": "failed-call", "purpose": "research"}}])
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] == 2
    assert metrics["research_model_usage"]["known_model_tokens"] == 5


@pytest.mark.parametrize("durations", [(1e308, 1e308), (10 ** 400, 0.5)])
def test_nonfinite_or_unrepresentable_duration_sum_is_unknown(tmp_path, durations):
    metrics = observe(tmp_path, [model(call_id=str(index), elapsed=value)
                                for index, value in enumerate(durations)])
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["research_model_wall_seconds_known_lower_bound"] is None
    assert metrics["research_model_wall_seconds_aggregation_unknown"] is True
