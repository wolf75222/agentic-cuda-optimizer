"""Whole-process restarts use a synthetic fixture here; real GPU proof is separate."""
import json

import pytest

from scripts.process_recovery_acceptance import run_acceptance


def test_actual_process_death_preserves_one_reward_and_original_files(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / "trials/one").mkdir(parents=True)
    (campaign / "evaluator_data").mkdir()
    (campaign / "evaluator_data/baseline.cu").write_text("fixture baseline")
    (campaign / "config.json").write_text(json.dumps({"workload": "scan"}))
    (campaign / "environment.json").write_text(json.dumps({"uuid": "synthetic-only"}))
    (campaign / "baseline-measurement.json").write_text(json.dumps({
        "ok": True, "contract_hash": "fixture-contract", "cases": {"small": {"latency_ms": 1.0}}}))
    (campaign / "trials/one/result.json").write_text(json.dumps({"tell": {
        "evaluation_id": "synthetic-measurement", "valid": True,
        "purpose": "tree_reward", "fresh": True, "latencies": {"small": .9},
        "state": {"candidate_id": "fixture-candidate", "source": "fixture source", "parameters": {}},
        "consumption": {"trials": 1, "gpu_seconds": 1000000}}}))
    receipt = run_acceptance(campaign, tmp_path / "proof")
    assert receipt["original_evidence_unchanged"]
    assert receipt["gpu_requests"] == receipt["model_calls"] == 0
    for boundary, result in receipt["cases"].items():
        assert result["crashed_exit_code"] == 87
        assert result["before_recovery"] == {"result_ready": 1, "reward_claims": 0}
        assert result["recovered"]["tree_reward_events"] == 1
        assert result["recovered"]["usage"] == {"trials": 3}
    with pytest.raises(ValueError, match="fresh functional"):
        run_acceptance(campaign, tmp_path / "proof")
