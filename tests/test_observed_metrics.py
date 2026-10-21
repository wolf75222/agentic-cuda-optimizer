"""CPU-only metadata regressions; no model, GPU, or scheduler activity."""
import pytest

from optimizer_agent.suite_orchestrator import observed_metrics, write_json


def test_unknown_metrics_use_computed_promotion_and_research_values(tmp_path):
    write_json(tmp_path / "generation/clock.json", {"started_at": 100.0})
    write_json(tmp_path / "generation/events.json", [
        {"kind": "model", "created_at": 200.0, "payload": {
            "call_id": "research-call", "purpose": "research",
            "elapsed_seconds": 301.5568711250089,
            "usage": {"input_tokens": 20, "output_tokens": 3}}},
        {"kind": "promotion", "created_at": 969.6877868175507},
    ])

    metrics = observed_metrics(tmp_path, {"usage": {}})

    assert metrics["first_useful_gain_status"] == "observed"
    assert metrics["time_to_first_useful_gain_seconds"] == pytest.approx(869.6877868175507)
    assert metrics["research_model_wall_seconds"] == 301.5568711250089
    assert metrics["unknown_metrics"] == []


def test_missing_metric_evidence_remains_unknown_instead_of_zero(tmp_path):
    metrics = observed_metrics(tmp_path, {"usage": {}})

    assert metrics["time_to_first_useful_gain_seconds"] is None
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["unknown_metrics"] == [
        "time_to_first_useful_gain_seconds", "research_model_wall_seconds"]
    assert "research_seconds" not in metrics


def test_recorded_zero_research_cost_is_known(tmp_path):
    write_json(tmp_path / "generation/events.json", [
        {"kind": "model", "created_at": 100.0, "payload": {
            "call_id": "proposal-call", "purpose": "proposal",
            "elapsed_seconds": 2.0, "usage": {"input_tokens": 4, "output_tokens": 1}}},
    ])

    metrics = observed_metrics(tmp_path, {"usage": {}})

    assert metrics["research_model_wall_seconds"] == 0
    assert metrics["research_model_tokens"] == 0
    assert metrics["unknown_metrics"] == ["time_to_first_useful_gain_seconds"]


def test_top_level_costs_do_not_replace_missing_computed_research_metric(tmp_path):
    metrics = observed_metrics(tmp_path, {
        "time_to_first_useful_gain_seconds": 2.0,
        "research_model_wall_seconds": 37.0,
        "research_seconds": 99.0,
        "usage": {"research_seconds": 101.0},
    })

    assert metrics["time_to_first_useful_gain_seconds"] == 2.0
    assert metrics["research_model_wall_seconds"] is None
    assert metrics["unknown_metrics"] == ["research_model_wall_seconds"]
