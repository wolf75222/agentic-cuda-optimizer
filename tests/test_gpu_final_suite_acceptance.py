"""CPU control-flow simulations; never interpreted as ROMEO/CUDA evidence."""
import json

from scripts import gpu_final_suite_acceptance as acceptance
from test_gpu_final_isolation_acceptance import SimulatedRunner


class SanitizedCPU(SimulatedRunner):
    def run_sanitizer(self, run_id, request, tool):
        return {"ok": True, "tool": tool, "evidence_kind": "simulated_cpu_test",
                "environment": self.get_device_properties(0)}


def prepared(tmp_path):
    runner = SanitizedCPU(tmp_path / "simulated-worker")
    recording = acceptance.FinalReceiptRunner(runner, tmp_path / "receipts")
    evaluator, baseline, candidate, _, _ = acceptance.prepare(tmp_path / "control", "cpu-final-suite",
        runner=recording, properties=runner.get_device_properties(0))
    return evaluator, baseline, candidate, recording


def test_plan_does_not_expose_final_or_contact_slurm(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU planning cannot contact Slurm")
    monkeypatch.setattr(acceptance, "SlurmBackend", forbidden)
    assert acceptance.main(["--output", str(tmp_path / "plan")]) == 0
    plan = json.loads((tmp_path / "plan/plan.json").read_text())
    assert plan["submitted"] is False and plan["model_calls"] == 0
    assert plan["search_shapes"] != plan["final_shapes"]
    assert not list((tmp_path / "plan/inputs").glob("final-*"))


def test_real_finalization_flow_uses_final_shapes_and_exposed_holdout_once(tmp_path):
    evaluator, baseline, candidate, recording = prepared(tmp_path)
    result = acceptance.check_final(evaluator, baseline, candidate, recording)
    assert result["ok"], result["checks"]
    assert result["final_benchmark_requests"] == 16
    assert len(recording.sanitizers) == 8


def test_check_rejects_legacy_timing_of_search_after_final_validation(tmp_path, monkeypatch):
    evaluator, baseline, candidate, recording = prepared(tmp_path)
    original = evaluator._confirm_suite

    def legacy_confirmation(candidate, incumbent, suite):
        # Reproduce the actual pre-444 error, with valid final correctness
        # receipts followed by timing on the old public performance suite.
        assert suite == "final"
        evaluator._validate(candidate, "performance")
        evaluator._validate(incumbent, "performance")
        return original(candidate, incumbent, "performance")

    monkeypatch.setattr(evaluator, "_confirm_suite", legacy_confirmation)
    result = acceptance.check_final(evaluator, baseline, candidate, recording)
    assert result["checks"]["final_oracles_passed"]
    assert not result["ok"]
    assert not result["checks"]["fresh_final_shapes_only"]
    assert not result["checks"]["confirmation_explicitly_final"]


def test_check_rejects_sanitizer_misrouting_despite_valid_final_labels(tmp_path, monkeypatch):
    evaluator, baseline, candidate, recording = prepared(tmp_path)
    original = recording.run_sanitizer

    def misrouted_request(run_id, request, tool):
        report = original(run_id, request, tool)
        recording.sanitizers[-1]["request"]["arguments"][-1]["value"] = 17
        return report

    monkeypatch.setattr(recording, "run_sanitizer", misrouted_request)
    result = acceptance.check_final(evaluator, baseline, candidate, recording)
    assert result["checks"]["all_four_sanitizers_on_final_cases"]
    assert not result["checks"]["sanitizer_requests_use_exact_final_inputs"]
    assert not result["ok"]
