"""Grouped studies freeze every method before exposing any common final data."""
import json
import time
from types import SimpleNamespace

import pytest

from optimizer_agent.campaign import CampaignRuntime
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate, LaunchRule


def test_generation_export_reconstructs_exact_artifact_without_final_access(tmp_path):
    source = tmp_path / "source.cu"
    source.write_text('extern "C" __global__ void candidate() {}')
    candidate = Candidate("frozen", source, "candidate", LaunchRule())

    def forbidden_final(*args, **kwargs):
        pytest.fail("Final evaluation must remain private until every method is frozen")

    runtime = CampaignRuntime.__new__(CampaignRuntime)
    runtime.directory = tmp_path
    runtime.config = CampaignConfig(run_id="grouped-study")
    runtime.generation_only = True
    runtime.started_at = time.time()
    runtime.contract_hash = "immutable-contract"
    runtime.environment = {"gpu_uuid": "reserved-device"}
    runtime.baseline_latencies = {"case": 1.0}
    runtime.evaluator = SimpleNamespace(finalize=forbidden_final)
    runtime.store = SimpleNamespace(
        incumbent=lambda run_id: {"candidate_id": "frozen"},
        budget_state=lambda run_id: {"spent": {"model_tokens": 23}, "reserved": {}},
        events=lambda run_id: [])
    runtime.candidate = lambda candidate_id: candidate
    result = runtime.finalize({"trials_done": 2})
    assert result["certification_status"] == "generation_complete"
    assert not (tmp_path / "holdout-exposed.json").exists()
    assert not (tmp_path / "final.json").exists()
    frozen = json.loads((tmp_path / "candidate.json").read_text())
    reconstructed = Candidate(frozen["candidate_id"], frozen["source"], frozen["kernel"],
                              LaunchRule(**frozen["launch_rule"]), frozen["parameters"])
    assert reconstructed.artifact_hash == candidate.artifact_hash == frozen["artifact_hash"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["usage"]["model_tokens"] is None
    assert manifest["usage"]["model_calls"] is None
    assert manifest["recorded_ledger_spent"]["model_tokens"] == 23
    assert manifest["model_usage_observation"]["unattributed_recorded_ledger_model_entries"] == {"model_tokens":23}
    assert manifest["final"] == {"status": "deferred_common_final_evaluation", "ok": False}


@pytest.mark.parametrize("saved_mode,requested_mode", [(False, True), (True, False)])
def test_resume_cannot_change_final_exposure_policy(tmp_path, saved_mode, requested_mode):
    (tmp_path / "execution-mode.json").write_text(json.dumps({"generation_only": saved_mode}))
    runner = SimpleNamespace(get_device_properties=lambda *_: pytest.fail("Policy drift must reject before GPU access"))
    with pytest.raises(ValueError, match="Finalization policy changed"):
        CampaignRuntime(CampaignConfig(), runner, tmp_path, generation_only=requested_mode)


def test_generation_cannot_reuse_final_exposed_directory(tmp_path):
    (tmp_path / "holdout-exposed.json").write_text("{}")
    with pytest.raises(ValueError, match="exposed final suite"):
        CampaignRuntime(CampaignConfig(), None, tmp_path, generation_only=True)
