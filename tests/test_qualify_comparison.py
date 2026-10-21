from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/qualify_comparison.py"
SPEC = importlib.util.spec_from_file_location("qualify_comparison_script", MODULE_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
from evaluation import Candidate, LaunchRule


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    config = helper.CampaignConfig(run_id="qualification-test", final_sizes=(17,),
                                   required_sanitizers=("memcheck", "racecheck", "initcheck", "synccheck"))
    directory = tmp_path / config.run_id
    direct_dir = directory / "direct-comparison"
    direct_dir.mkdir(parents=True)
    candidates = {}
    for mode in helper.MODES:
        path = direct_dir / (mode + ".cu")
        path.write_text('extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n){}\n//' + mode)
        candidates[mode] = Candidate(mode + "-generated", path, "matmul", LaunchRule())
    old = helper.direct_evaluator(replace(config, required_sanitizers=()), directory, None)
    properties = {"name": "Test CUDA", "uuid": "physical-GPU", "architecture": "sm_90",
                  "driver_version": 13010, "nvrtc_version": "12.6", "cuda_toolkit_version": 12060,
                  "harness_build_id": "exact-source-and-compiler"}
    direct = {"ok": True, "status": "not_improved", "contract_hash": old.contract_hash,
              "protocol": asdict(old.contract), "environment": properties,
              "pairs": [{"case_id": "final-17", "series": 0, "order": ["vanilla", "agent"],
                         "measurements": {"agent": {"mean_ms": 1}, "vanilla": {"mean_ms": 1}}}],
              "interval": {"gain": 1, "lower": .99, "upper": 1.01},
              "candidate_artifacts": {mode: candidate.artifact_hash for mode, candidate in candidates.items()}}
    summary = {"job_id": "770104", "hardware": {"device_properties": properties},
               "generated_agent_vs_vanilla": direct, "ok": True, "status": "completed",
               "results": {mode: {"status": "validated_no_gain"} for mode in helper.MODES}}
    helper.write_json(directory / "comparison.json", summary)
    helper.write_json(direct_dir / "comparison.json", direct)
    helper.write_json(directory / "worker-terminal.json", {"job_id": "770104", "state": "COMPLETED",
                      "exit_code": "0:0", "terminal": True})
    calls = {"backends": [], "qualifications": []}
    class Backend:
        def __init__(self, *args, **kwargs):
            self.job_id = None
            self.closed = False
            self.properties = dict(properties)
            calls["backends"].append(self)
        def start(self):
            self.job_id = "770200"
            return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return SimpleNamespace(get_device_properties=lambda device: self.properties)
        def close(self, *, cancel=False):
            self.closed = True
        def wait_until_finished(self, timeout):
            return {"job_id": self.job_id, "state": "COMPLETED", "exit_code": "0:0", "terminal": True}
    monkeypatch.setattr(helper, "SlurmBackend", Backend)
    monkeypatch.setattr(helper, "load_direct_candidates", lambda config, directory: candidates)
    real_evaluator = helper.direct_evaluator
    def evaluator(config, directory, runner):
        result = real_evaluator(config, directory, runner)
        def qualify(candidate, suite):
            assert suite == "performance"
            calls["qualifications"].append(candidate.candidate_id)
            cases = result._suites[suite]
            return {"ok": True, "status": "sanitizer_qualified", "artifact_hash": candidate.artifact_hash,
                    "contract_hash": result.contract_hash, "required_sanitizers": list(config.required_sanitizers),
                    "validation": {"ok": True, "cases": [{"case_id": case.case_id} for case in cases]},
                    "sanitizers": [{"ok": True, "tool": tool, "case_id": case.case_id}
                                   for tool in config.required_sanitizers for case in cases]}
        result.qualify_sanitizers = qualify
        result.benchmark = lambda *args, **kwargs: pytest.fail("qualification must never benchmark")
        result.confirm = lambda *args, **kwargs: pytest.fail("qualification must never replace paired timings")
        return result
    monkeypatch.setattr(helper, "direct_evaluator", evaluator)
    return SimpleNamespace(config=config, directory=directory, direct=direct, summary=summary,
                           candidates=candidates, properties=properties, calls=calls)


def test_posthoc_qualification_preserves_measurements_and_binds_both_exact_artifacts(campaign):
    result = helper.qualify_comparison(campaign.config, campaign.directory, "new-qualification",
                                      controller_note="Actual measurement controller predates the final precision guard")
    assert result["ok"]
    assert result["main_measurement_job_id"] == "770104"
    assert result["qualification_job_id"] == "770200"
    assert result["measurement_contract_hash"] == campaign.direct["contract_hash"]
    assert result["qualification_contract_hash"] != result["measurement_contract_hash"]
    assert result["input_hashes"] == helper.input_hashes(campaign.directory)
    assert campaign.calls["qualifications"] == ["agent-generated", "vanilla-generated"]
    assert campaign.calls["backends"][0].closed
    direct = helper.read_json(campaign.directory / "direct-comparison/comparison.json")
    summary = helper.read_json(campaign.directory / "comparison.json")
    standalone = helper.read_json(campaign.directory / result["proof_reference"])
    assert direct["pairs"] == campaign.direct["pairs"]
    assert direct["contract_hash"] == campaign.direct["contract_hash"]
    assert direct["protocol"] == json.loads(json.dumps(campaign.direct["protocol"]))
    assert standalone == direct["sanitizer_qualification"] == summary["generated_agent_vs_vanilla"]["sanitizer_qualification"]
    assert summary["ok"] and summary["status"] == "completed"


def test_missing_terminal_main_job_never_allocates(campaign):
    helper.write_json(campaign.directory / "worker-terminal.json", {"job_id": "770104", "terminal": False})
    with pytest.raises(ValueError, match="successful terminal accounting"):
        helper.qualify_comparison(campaign.config, campaign.directory, "new")
    assert not campaign.calls["backends"]


def test_mismatched_candidate_artifact_never_allocates(campaign):
    for path in (campaign.directory / "comparison.json", campaign.directory / "direct-comparison/comparison.json"):
        result = helper.read_json(path)
        direct = result.get("generated_agent_vs_vanilla", result)
        direct["candidate_artifacts"]["agent"] = "changed-artifact"
        helper.write_json(path, result)
    with pytest.raises(ValueError, match="do not match the paired measurement"):
        helper.qualify_comparison(campaign.config, campaign.directory, "new")
    assert not campaign.calls["backends"]


def test_wrong_physical_gpu_fails_and_closes_exact_qualification_job(campaign, monkeypatch):
    original = helper.SlurmBackend
    def backend(*args, **kwargs):
        value = original(*args, **kwargs)
        value.properties["uuid"] = "another-physical-GPU"
        return value
    monkeypatch.setattr(helper, "SlurmBackend", backend)
    with pytest.raises(RuntimeError, match="Qualification failed"):
        helper.qualify_comparison(campaign.config, campaign.directory, "new")
    proof = helper.read_json(campaign.directory / "direct-comparison/sanitizer-qualification.json")
    assert "UUID/toolchain differs" in proof["error"]
    assert not proof["ok"] and not campaign.calls["qualifications"]
    assert campaign.calls["backends"][0].closed


def test_changed_fixed_inputs_fail_before_allocating(campaign, monkeypatch):
    original = helper.direct_evaluator
    def evaluator(*args, **kwargs):
        value = original(*args, **kwargs)
        path = next((campaign.directory / "direct-comparison/inputs").rglob("a.bin"))
        path.write_bytes(path.read_bytes() + b"changed")
        return value
    monkeypatch.setattr(helper, "direct_evaluator", evaluator)
    with pytest.raises(ValueError, match="inputs changed"):
        helper.qualify_comparison(campaign.config, campaign.directory, "new")
    assert not campaign.calls["backends"]


def test_non_sanitizer_protocol_change_is_rejected(campaign):
    for path in (campaign.directory / "comparison.json", campaign.directory / "direct-comparison/comparison.json"):
        result = helper.read_json(path)
        result.get("generated_agent_vs_vanilla", result)["protocol"]["warmup"] = 999
        helper.write_json(path, result)
    with pytest.raises(ValueError, match="changes more than"):
        helper.qualify_comparison(campaign.config, campaign.directory, "new")
    assert not campaign.calls["backends"]


def test_completed_qualification_is_reused_without_another_allocation(campaign):
    first = helper.qualify_comparison(campaign.config, campaign.directory, "first")
    second = helper.qualify_comparison(campaign.config, campaign.directory, "unused-second")
    assert first == second
    assert len(campaign.calls["backends"]) == 1
