"""A reviewed negative finding must precede FINAL oracle/sanitizer/ranking calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from optimizer_agent.campaign import Candidate, LaunchRule
from optimizer_agent.suite_orchestrator import candidate_receipt, evaluate_group
from test_final_group_identity import fixture


def blocked_fixture(tmp_path):
    entries, generations, runner = fixture(tmp_path)
    entry = next(entry for entry in entries if entry["mode"] == "no_autotuning")
    path = tmp_path / "reviewed-blocked-fixture.cu"
    path.write_text("// CPU fixture of a separately reviewed disqualified artifact\n" +
                    (tmp_path / "no_autotuning.cu").read_text())
    candidate = Candidate("reviewed-bad", path, "matmul", LaunchRule(kind="square_2d", block=(16, 16, 1)))
    generations[entry["config"]["run_id"]] = {
        "candidate": candidate_receipt(candidate), "method_status": "completed"}
    return entries, generations, runner, candidate


@pytest.mark.parametrize("uncertain", [False, True])
def test_exact_bad_artifact_is_frozen_but_never_executed_or_ranked(tmp_path, uncertain):
    entries, generations, runner, bad = blocked_fixture(tmp_path)
    original = deepcopy(generations)
    incident = {"incident_id": "reviewed-static-violation", "artifact_hash": bad.artifact_hash,
                "scientific_qualification_eligible": False}
    mapping = {} if uncertain else {bad.artifact_hash: [incident]}
    observations = [{"status": "uncertain", "artifact_hash": bad.artifact_hash,
                     "qualification_status": "not_admitted"}] if uncertain else []
    output = tmp_path / "final"
    result = evaluate_group(entries, generations, runner, output, 12345, "c" * 64,
        artifact_qualification_incidents=mapping, artifact_incident_observations=observations)
    assert result["evaluation_finished"] is True
    assert result["all_modes_available"] is True
    assert result["artifact_qualification_excluded_modes"] == ["no_autotuning"]
    blocked = result["candidates"]["no_autotuning"]
    assert blocked["qualified"] is False and blocked["scientific_qualification_eligible"] is False
    assert blocked["generation_method_status"] == "completed"
    assert blocked["status"] == ("artifact_qualification_uncertain" if uncertain else "artifact_scientifically_disqualified")
    assert "confirmation" not in blocked and "sanitizer_qualification" not in blocked
    assert "full_vs_no_autotuning" not in result["method_comparisons"]
    assert result["candidates"]["full"]["qualified"] is True
    assert result["candidates"]["full"]["status"] == "validated_no_gain"
    assert "no_autotuning" in json.loads((output / "frozen-candidates.json").read_text())
    assert not (output / "qualification/no_autotuning.json").exists()
    forbidden = hashlib.sha256(bad.source.read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(request.source).read_bytes()).hexdigest() != forbidden
               for request in runner.calls)
    assert all(hashlib.sha256(Path(request.source).read_bytes()).hexdigest() != forbidden
               for request, tool in runner.sanitizer_calls)
    assert generations == original  # Raw generation/outcomes are never rewritten.
    assert blocked["final_closure"]["status"] == "closed"
    assert blocked["final_closure"]["candidate_request_cost"]["observed_rpc_calls"] == 0
    from optimizer_agent.quality_suite import summarize_suite
    # Synthetic closed-execution fixture only: this test observes no Slurm job.
    closed_fixture = {**result, "execution_complete":True}
    summary = summarize_suite({"entries":entries, "modes":sorted({e["mode"] for e in entries}),
        "workloads":["matmul"]}, generations, {result["group_id"]:closed_fixture},
        artifact_qualification_incidents=mapping, artifact_incident_observations=observations)
    assert summary["complete"] is True and summary["scientific_qualified"] is False


def test_missing_source_remains_unavailable_instead_of_becoming_a_refused_available_artifact(tmp_path):
    entries, generations, runner, bad = blocked_fixture(tmp_path)
    bad.source.unlink()
    result = evaluate_group(entries, generations, runner, tmp_path / "final", 12345, "c" * 64)
    assert result["evaluation_finished"] is True
    assert result["all_modes_available"] is False
    assert result["artifact_qualification_excluded_modes"] == []
    assert result["candidates"]["no_autotuning"]["status"] == "generation_unavailable"


@pytest.mark.parametrize("global_unknown", [False, True])
def test_baseline_or_unresolved_identity_blocks_before_heldout_inputs(tmp_path, global_unknown):
    entries, generations, runner = fixture(tmp_path)
    baseline_artifact = next(iter(generations.values()))["candidate"]["artifact_hash"]
    incidents = {} if global_unknown else {baseline_artifact: [{"scientific_qualification_eligible": False}]}
    observations = [{"status": "uncertain", "artifact_hash": None}] if global_unknown else []
    output = tmp_path / "refused-final"
    with pytest.raises(ValueError, match="baseline or global artifact incident"):
        evaluate_group(entries, generations, runner, output, 12345, "c" * 64,
            artifact_qualification_incidents=incidents, artifact_incident_observations=observations)
    assert runner.calls == runner.sanitizer_calls == []
    assert not (output / "holdout-reveal.json").exists()
    # Constructing WorkloadSuite creates an empty directory; no buffers are made.
    assert not list((output / "sealed-inputs").rglob("*.bin"))
