"""CPU-only final-group identity controls, without CUDA/Slurm/model execution."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from optimizer_agent.campaign import Candidate, LaunchRule
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite, summarize_suite
from optimizer_agent.suite_orchestrator import REPO, _hash, candidate_receipt, evaluate_group


class CPUControlRunner:
    def __init__(self, directory, *, sanitizer_status="passed", incorrect=False):
        self.directory = directory
        self.calls, self.sanitizer_calls = [], []
        self.sanitizer_status, self.incorrect = sanitizer_status, incorrect

    def get_device_properties(self, device):
        return {"uuid":"CPU-CONTROL-NOT-A-GPU", "architecture":"sm_90", "driver_version":12000,
                "nvrtc_version":"12.0", "cuda_toolkit_version":12000,"harness_build_id":"cpu-control"}

    def run(self, run_id, request):
        self.calls.append(request)
        n = request.arguments[-1].value
        a, b = [np.fromfile(argument.input_file,dtype="<f4").reshape(n,n) for argument in request.arguments[:2]]
        result = (a.astype(np.float64) @ b.astype(np.float64)).astype("<f4")
        if self.incorrect:
            result.flat[0] = np.nan
        output = self.directory / (run_id + ".bin")
        result.tofile(output)
        return {"ok":True,"environment":self.get_device_properties(0),"guard_readonly_ok":True,
                "guard_outputs_poisoned":True,"outputs":[{"path":str(output),"bytes":result.nbytes}],
                "latency_samples_ms":[1.0]*request.iterations}

    def run_sanitizer(self, run_id, request, tool):
        self.sanitizer_calls.append((request,tool))
        return {"ok":self.sanitizer_status == "passed", "status":self.sanitizer_status,
                "tool":tool,"environment":self.get_device_properties(0)}


def fixture(tmp_path, *, path_sensitive=False, **runner_options):
    source_bytes = (REPO/"examples/naive_matmul.cu").read_bytes()
    source_path = REPO/"examples/naive_matmul.cu"
    if path_sensitive:
        source_bytes = b"#define SOURCE_LOCATION __FILE__\n" + source_bytes
        source_path = tmp_path/"path-sensitive-baseline.cu"
        source_path.write_bytes(source_bytes)
    config = CampaignConfig(run_id="cpu-identity",search_sizes=(4,),profile=False,
                            benchmark_series=1,confirmation_batches=2,launches_per_series=1,baseline_source=str(source_path))
    plan = plan_suite(config,seeds=(1,2),include_greedy=True)
    modes = {"full","no_autotuning","no_research","pmpp_permanent_only","no_pmpp","greedy_parent"}
    entries = [entry for entry in plan["entries"] if entry["workload"] == "matmul" and entry["seed"] == 1 and entry["mode"] in modes]
    assert len(entries) == 6
    generations = {}
    for entry in entries:
        source = tmp_path / (entry["mode"]+".cu")
        source.write_bytes(source_bytes)
        candidate = Candidate("baseline",source,"matmul",LaunchRule(kind="square_2d",block=(16,16,1)))
        generations[entry["config"]["run_id"]] = {"candidate":candidate_receipt(candidate)}
    return entries,generations,CPUControlRunner(tmp_path,**runner_options)


def run(entries,generations,runner,directory):
    seed = 12345  # Local fixture seed, unrelated to any real study.
    return evaluate_group(entries,generations,runner,directory,seed,hashlib.sha256(str(seed).encode()).hexdigest())


def test_identical_baseline_methods_share_one_qualification_without_paired_noise(tmp_path):
    entries,generations,runner = fixture(tmp_path)
    directory = tmp_path/"final"
    result = run(entries,generations,runner,directory)
    case_count = len(result["shapes"])
    assert len(runner.sanitizer_calls) == 4*case_count
    assert sum(request.mode == "run" for request in runner.calls) == 2*case_count+2
    assert sum(request.mode == "benchmark" for request in runner.calls) == case_count
    original = json.loads((directory/"qualification/baseline.json").read_text())
    for entry in entries:
        mode = entry["mode"]
        row = result["candidates"][mode]
        assert row["qualified"] and row["status"] == "validated_no_gain"
        assert row["gain_basis"] == "exact_baseline_artifact"
        qualification = row["sanitizer_qualification"]
        assert qualification["qualification_execution"] == "alias_of_same_group_evidence"
        assert qualification["fresh_gpu_runs"] == 0
        assert qualification["evaluation_id"] == original["evaluation_id"]
        assert qualification["qualification_evidence_ref"]["sha256"] == hashlib.sha256((directory/"qualification/baseline.json").read_bytes()).hexdigest()
        assert qualification["qualification_evidence_ref"]["canonical_sha256"] == _hash(original)
        assert qualification["qualification_evidence_ref"]["path"] == "qualification/baseline.json"
        assert "sanitizers" not in qualification and "validation" not in qualification
    assert result["method_comparisons"]
    for comparison in result["method_comparisons"].values():
        assert comparison["status"] == "exact_artifact_identity"
        assert comparison["pairs"] == [] and "interval" not in comparison
        assert comparison["paired_measurement_performed"] is False
    # The memo is confined to the call: another fresh group must qualify anew.
    second = run(entries,generations,runner,tmp_path/"independent-final")
    assert len(runner.sanitizer_calls) == 8*case_count
    assert second["baseline_qualification"]["evaluation_id"] != original["evaluation_id"]


@pytest.mark.parametrize("options,status", [({"sanitizer_status":"tool_unavailable"},"sanitizer_qualification_incomplete"),
                                            ({"incorrect":True},"sanitizer_qualification_failed")])
def test_failed_or_incomplete_shared_qualification_never_becomes_success(tmp_path,options,status):
    entries,generations,runner = fixture(tmp_path,**options)
    directory = tmp_path/"final"
    result = run(entries,generations,runner,directory)
    original = result["baseline_qualification"]
    assert not original["ok"] and original["status"] == status
    assert len(runner.sanitizer_calls) == (0 if options.get("incorrect") else 1)
    assert not any(request.mode == "benchmark" for request in runner.calls)
    assert not result["method_comparisons"]
    for row in result["candidates"].values():
        alias = row["sanitizer_qualification"]
        assert not row["qualified"] and not alias["ok"]
        assert alias["status"] == status and alias["evaluation_id"] == original["evaluation_id"]
        assert alias["fresh_gpu_runs"] == 0
        assert alias["qualification_evidence_ref"]["sha256"] == hashlib.sha256((directory/"qualification/baseline.json").read_bytes()).hexdigest()
        assert alias["qualification_evidence_ref"]["canonical_sha256"] == _hash(original)


def test_path_sensitive_same_source_is_freshly_qualified_without_derived_gain_one(tmp_path):
    entries,generations,runner = fixture(tmp_path,path_sensitive=True)
    result = run(entries,generations,runner,tmp_path/"final")
    assert len(runner.sanitizer_calls) == (len(entries)+1)*4*len(result["shapes"])
    for row in result["candidates"].values():
        assert row["qualified"] and row["status"] == "final_comparison_identity_unproven"
        assert row["gain_basis"] == "unproven_artifact_context"
        assert row["sanitizer_qualification"]["qualification_execution"] == "fresh"
        assert "gain" not in row and "confirmation" not in row
    for comparison in result["method_comparisons"].values():
        assert not comparison["ok"] and comparison["status"] == "artifact_identity_context_unproven"
        assert comparison["pairs"] == [] and "interval" not in comparison
    summary = summarize_suite({"entries":entries,"modes":[entry["mode"] for entry in entries],"workloads":["matmul"]},
                              generations,{entries[0]["group_id"]:result})
    assert all(row["gain"] is None and row["final_qualified"] for row in summary["rows"])
    costs = [row["final_closure"]["candidate_request_cost"] for row in result["candidates"].values()]
    assert all(cost["unique_request_ids"] > 0 for cost in costs)
    shared = result["final_cost_observation"]["shared_group_request_cost"]
    assert shared["unique_request_ids"] + sum(cost["unique_request_ids"] for cost in costs) == len(runner.calls) + len(runner.sanitizer_calls)


def test_no_gain_closes_final_independently_with_shared_costs_and_partial_generation_snapshot(tmp_path):
    entries, generations, runner = fixture(tmp_path)
    for generation in generations.values():
        generation.update(task_started_at_unix=time.time() - 100, usage={"model_tokens": 13, "model_calls": 7},
                          method_status="failed")
    directory = tmp_path / "final"
    result = run(entries, generations, runner, directory)
    evidence = json.loads((directory / "final-cost-observation.json").read_text())
    assert evidence["kind"] == "final_group_request_cost_observation"
    shared = evidence["request_costs"]["shared"]
    assert shared["unique_request_ids"] == len(runner.calls) + len(runner.sanitizer_calls)
    assert shared["observed_rpc_calls"] == shared["unique_request_ids"]
    assert shared["reported_native_reservation_seconds"] is None
    assert shared["native_duration_missing_request_ids"] == shared["request_ids"]
    for entry in entries:
        candidate = result["candidates"][entry["mode"]]
        closure = candidate["final_closure"]
        assert closure["status"] == "closed" and closure["outcome"] == "validated_no_gain"
        assert candidate["generation_method_status"] == "failed"  # Valid artifact is not method success.
        assert closure["seconds_since_generation_task_start"] >= 100
        assert closure["reported_generation_usage"] == {"model_tokens": 13, "model_calls": 7, "gpu_seconds": None}
        assert closure["final_model_tokens"] is None and closure["final_model_calls"] is None and closure["cost_usd"] is None
        assert closure["candidate_request_cost"]["unique_request_ids"] == 0  # Exact aliases generated no fresh RPCs.
        assert closure["candidate_request_cost"]["reported_native_reservation_seconds"] is None
        assert closure["shared_group_request_cost"] == shared
        assert closure["cost_evidence"]["sha256"] == hashlib.sha256((directory / "final-cost-observation.json").read_bytes()).hexdigest()
        assert "time_to_independently_confirmed_final_gain_seconds" not in candidate
    public_trace = json.dumps(evidence)
    assert "input_file" not in public_trace and "outputs" not in public_trace and "arguments" not in public_trace
    assert "final_seed" not in public_trace and "sealed-inputs" not in public_trace


def test_numerical_failure_is_closed_without_becoming_qualified_or_gained(tmp_path):
    entries, generations, runner = fixture(tmp_path, incorrect=True)
    result = run(entries, generations, runner, tmp_path / "final")
    for candidate in result["candidates"].values():
        assert not candidate["qualified"]
        assert candidate["final_closure"]["status"] == "closed"
        assert candidate["final_closure"]["outcome"] == "baseline_validation_failed"
        assert candidate["final_closure"]["seconds_since_generation_task_start"] is None
        assert candidate["final_closure"]["reported_generation_usage_source"] == "unreported"


def test_transport_failure_returned_after_qualification_does_not_create_a_final_closure(tmp_path, monkeypatch):
    import optimizer_agent.campaign as campaign
    entries, generations, runner = fixture(tmp_path)
    monkeypatch.setattr(campaign.Evaluator, "benchmark", lambda self, candidate: {
        "ok": False, "error": "ssh: connect to host example.invalid port 22: Connection timed out"})
    directory = tmp_path / "final"
    result = run(entries, generations, runner, directory)
    assert result["infrastructure_interrupted"] and not result["evaluation_finished"]
    assert all("final_closure" not in candidate for candidate in result["candidates"].values())
    assert not (directory / "final-cost-observation.json").exists()


@pytest.mark.parametrize("failure", [{"ok": False, "status": "infrastructure_failure"},
                                      {"ok": False, "status": "tool_unavailable", "worker_poisoned": True}])
def test_explicit_infrastructure_or_poisoned_worker_never_creates_a_closure_or_retry_permission(tmp_path, failure):
    entries, generations, runner = fixture(tmp_path)
    runner.run_sanitizer = lambda run_id, request, tool: {
        "environment": runner.get_device_properties(0), "tool": tool, **failure}
    directory = tmp_path / "final"
    result = run(entries, generations, runner, directory)
    assert result["infrastructure_interrupted"] and not result["evaluation_finished"]
    assert result["infrastructure_category"] == "final_request_infrastructure_failure"
    assert all("final_closure" not in candidate for candidate in result["candidates"].values())
    assert not (directory / "final-cost-observation.json").exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_native_cost_deduplicates_replayed_physical_ids_and_preserves_conflicts(tmp_path, monkeypatch, conflict):
    import optimizer_agent.campaign as campaign
    entries, generations, _ = fixture(tmp_path)
    class Runner:
        count = 0
        def get_device_properties(self, device):
            return {"uuid": "CPU-CONTROL-NOT-A-GPU"}
        def run(self, run_id, request):
            self.count += 1
            return {"ok": True, "request_fingerprint": "a" * 64,
                    "reservation_wall_seconds": 2 if not conflict or self.count == 1 else 3,
                    "compile_ms": 500}
    class Evaluator:
        contract_hash = "contract-fixture"
        def __init__(self, runner, *args):
            self.runner = runner
        def qualification_context(self, candidate, suite):
            return {"key": "same-baseline-context"}
        def qualify_sanitizers(self, candidate, suite):
            self.runner.run("same-physical-request", object())
            self.runner.run("same-physical-request", object())
            return {"evaluation_id": "fixture-eval", "artifact_hash": candidate.artifact_hash,
                    "contract_hash": self.contract_hash, "suite": suite, "ok": True, "status": "passed"}
        def benchmark(self, candidate):
            self.runner.run("same-physical-request", object())
            return {"ok": True}
        def confirm(self, candidate, baseline):
            return {"ok": True, "status": "exact_artifact_identity"}
    monkeypatch.setattr(campaign, "Evaluator", Evaluator)
    directory = tmp_path / "final"
    result = run(entries, generations, Runner(), directory)
    shared = result["final_cost_observation"]["shared_group_request_cost"]
    assert shared["unique_request_ids"] == 1 and shared["observed_rpc_calls"] == 3
    assert shared["reported_compile_seconds"] == .5  # Replays never multiply native compilation.
    assert shared["reported_native_reservation_seconds"] == (None if conflict else 2)
    assert shared["native_duration_conflict_request_ids"] == (["same-physical-request"] if conflict else [])
    observation = json.loads((directory / "final-cost-observation.json").read_text())
    assert len(observation["observations"]) == 3
    assert all(row["candidate_request_cost"]["unique_request_ids"] == 0 for row in
               (candidate["final_closure"] for candidate in result["candidates"].values()))
