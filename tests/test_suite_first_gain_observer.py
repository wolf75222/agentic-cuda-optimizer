"""Coordinator-only observer tests: local files and fakes, no GPU/model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer_agent.campaign import Candidate,LaunchRule
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent import suite_orchestrator as suite
from optimizer_agent import codex_backend,upstream_original,vanilla_gpu_service


METRICS={"time_to_first_useful_gain_seconds":3.25,"model_tokens_to_first_useful_gain":None,
    "first_useful_gain_status":"observed","observer_wall_seconds":.25,
    "observer_gpu_requests":14,"first_useful_gain_scope":"Search observer only; no method feedback"}


def entry(mode):
    config=CampaignConfig(run_id="observer-coordinator",workload="matmul",experimental_mode="test",
        research_required=False,web_search_required=False)
    return {"mode":mode,"config":config.to_dict(),"workload":"matmul","seed":1729,"group_id":"observer-group"}


def frozen_candidate(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/"kernel.cu"
    source.write_bytes((suite.REPO/"examples/naive_matmul.cu").read_bytes())
    return suite.candidate_receipt(Candidate("frozen-observer-candidate",source,"matmul",
        LaunchRule(kind="square_2d",block=(16,16,1))))


def native_receipt(directory,*,status="completed"):
    candidate=frozen_candidate(Path(directory)/"evidence")
    receipt={"status":status,"candidate":candidate,"started_at":1000.0,"ended_at":1010.0,
        "usage":{"input_tokens":50,"output_tokens":7},"usage_events":[{"turn_id":"one-turn"}],
        "metrics":{"method_specific_count":4}}
    suite.write_json(Path(directory)/"evidence/receipt.json",receipt)
    suite.write_json(Path(directory)/"evidence/candidate.json",candidate)
    return receipt


def forbid_new_native_activity(monkeypatch):
    monkeypatch.setattr(suite,"_script",lambda name:pytest.fail("Completed native receipt must not restart the CLI/prompt"))
    monkeypatch.setattr(vanilla_gpu_service,"VanillaGPUService",lambda *args,**kwargs:pytest.fail("Completed receipt must not construct a GPU service"))
    monkeypatch.setattr(codex_backend,"default_codex_executable",lambda:pytest.fail("No model executable is needed to read durable receipts"))


def test_upstream_adapter_requests_parent_only_observation_and_preserves_metrics(tmp_path,monkeypatch):
    candidate=frozen_candidate(tmp_path/"original-artifact")
    calls=[]
    borrowed=object()
    def original(config,runner,directory,*,resume,finalize,observe_first_gain):
        calls.append({"config":config,"runner":runner,"directory":directory,"resume":resume,
                      "finalize":finalize,"observe_first_gain":observe_first_gain})
        return {"candidate":candidate,"metrics":dict(METRICS),"usage":{"input_tokens":20,"output_tokens":5}}
    monkeypatch.setattr(upstream_original,"run_upstream_campaign",original)
    result=suite.generation_adapter(entry("true_upstream"),borrowed,tmp_path/"original","session-original",resume=True)
    assert len(calls) == 1 and calls[0]["runner"] is borrowed
    assert calls[0]["observe_first_gain"] is True and calls[0]["finalize"] is False and calls[0]["resume"] is True
    assert result["metrics"] == METRICS
    assert result["candidate"]["artifact_hash"] == candidate["artifact_hash"]
    assert result["generation_scores_excluded_from_suite_final"] is True


def test_native_observer_metrics_are_collected_after_cli_return_and_persisted_outside_workspace(tmp_path,monkeypatch):
    directory=tmp_path/"native"
    borrowed=object()
    state={"finished":False,"metric_calls":0,"events":[]}
    class ParentService:
        def __init__(self,config,protected,workspace,session_id,*,repo_dir,runner,
                     observe_first_gain,observer_started_at_unix):
            assert runner is borrowed and observe_first_gain is True
            assert observer_started_at_unix == 999.0
            assert session_id == "session-native"
            assert Path(protected) == directory/"evidence" and Path(workspace) == directory/"workspace"
            state["events"].append("service_created")
        def observer_metrics(self):
            assert state["finished"],"Observer metrics must not be fed through the active model driver"
            state["metric_calls"] += 1
            state["events"].append("parent_observer_snapshot")
            return dict(METRICS)
    def driver(config,output,session_id,executable,*,service_factory,sensitive_root):
        assert executable == Path("/never-executed/codex")
        assert sensitive_root == tmp_path
        # The driver receives its normal service factory; no metric callback or
        # observer result is added to the model/prompt interface by coordinator.
        service_factory(config,Path(output)/"evidence",Path(output)/"workspace",session_id,repo_dir=suite.REPO)
        receipt=native_receipt(output)
        assert state["metric_calls"] == 0
        state["finished"]=True
        state["events"].append("cli_finished")
        return receipt
    monkeypatch.setattr(vanilla_gpu_service,"VanillaGPUService",ParentService)
    monkeypatch.setattr(codex_backend,"default_codex_executable",lambda:Path("/never-executed/codex"))
    monkeypatch.setattr(suite,"_script",lambda name:SimpleNamespace(run_vanilla=driver) if name == "run_codex_vanilla" else pytest.fail(name))
    monkeypatch.setattr(suite,"time",SimpleNamespace(time=lambda:999.0))
    result=suite.generation_adapter(entry("native_codex_vanilla"),borrowed,directory,"session-native",sensitive_root=tmp_path)
    assert state["events"] == ["service_created","cli_finished","parent_observer_snapshot"]
    assert state["metric_calls"] == 1
    assert json.loads((directory/"observer-metrics.json").read_text()) == METRICS
    assert not (directory/"workspace/observer-metrics.json").exists()
    assert result["metrics"]["method_specific_count"] == 4
    assert result["metrics"]["time_to_first_useful_gain_seconds"] == 3.25
    assert result["metrics"]["model_tokens_to_first_useful_gain"] is None
    assert result["usage"]["model_tokens"] == 57  # Final usage must not fill unknown usage at first gain.
    assert suite.load_candidate(result["candidate"]).artifact_hash == result["candidate"]["artifact_hash"]


def test_fresh_native_requires_protected_root_before_service_or_prompt(tmp_path,monkeypatch):
    forbid_new_native_activity(monkeypatch)
    directory=tmp_path/"never-started-native"
    with pytest.raises(ValueError,match="explicit protected study root"):
        suite.generation_adapter(entry("native_codex_vanilla"),object(),directory,"blocked-session")
    assert not directory.exists()


def test_completed_native_receipt_reuses_parent_snapshot_without_cli_or_service(tmp_path,monkeypatch):
    receipt=native_receipt(tmp_path)
    suite.write_json(tmp_path/"observer-metrics.json",METRICS)
    forbid_new_native_activity(monkeypatch)
    result=suite.generation_adapter(entry("native_codex_vanilla"),object(),tmp_path,"existing-session",resume=True)
    assert result["metrics"] == {"method_specific_count":4,**METRICS}
    assert result["candidate"]["artifact_hash"] == receipt["candidate"]["artifact_hash"]
    assert result["usage"]["model_tokens"] == 57 and result["generation_complete"] is True


def test_completed_cli_before_parent_snapshot_recovers_only_durable_observer_files(tmp_path,monkeypatch):
    receipt=native_receipt(tmp_path)
    artifact=receipt["candidate"]["artifact_hash"]
    state={"identity":{"method":"native_codex","contract_hash":"common-search-contract",
        "baseline_artifact_hash":"naive-reference","campaign_started_at_unix":1000.0},
        "observations":{artifact:{"status":"completed","artifact_hash":artifact,"started_at_unix":1002.0,
            "observer_wall_seconds":.25,"observer_gpu_requests":14,
            "consumption_at_observation":{"model_tokens":None,"model_calls":None,"method_gpu_evaluation_wall_seconds":.5}}},
        "first_useful_gain":{"artifact_hash":artifact,"elapsed_seconds":3.25,
            "consumption":{"model_tokens":None,"model_calls":None}}}
    suite.write_json(tmp_path/"evidence/gain-observer/summary.json",state)
    assert not (tmp_path/"observer-metrics.json").exists()
    forbid_new_native_activity(monkeypatch)
    result=suite.generation_adapter(entry("native_codex_vanilla"),object(),tmp_path,"existing-session",resume=True)
    assert result["metrics"]["time_to_first_useful_gain_seconds"] == 3.25
    assert result["metrics"]["model_tokens_to_first_useful_gain"] is None
    assert result["metrics"]["first_useful_gain_status"] == "observed"
    assert result["metrics"]["observer_confirmations_completed"] == 1
    assert "subsequent method consumption is unknown" in result["metrics"]["consumption_scope"]
    assert json.loads((tmp_path/"observer-metrics.json").read_text())["time_to_first_useful_gain_seconds"] == 3.25


def test_completed_cli_with_uncertain_observation_stays_unknown_without_replaying_gpu_or_prompt(tmp_path,monkeypatch):
    receipt=native_receipt(tmp_path)
    artifact=receipt["candidate"]["artifact_hash"]
    suite.write_json(tmp_path/"evidence/gain-observer/summary.json",{
        "observations":{artifact:{"status":"pending","started_at_unix":1002.0,
            "consumption_before_observer":{"model_tokens":None,"model_calls":None}}},
        "first_useful_gain":None})
    forbid_new_native_activity(monkeypatch)
    result=suite.generation_adapter(entry("native_codex_vanilla"),object(),tmp_path,"existing-session",resume=True)
    assert result["metrics"]["first_useful_gain_status"] == "unknown_incident"
    assert result["metrics"]["time_to_first_useful_gain_seconds"] is None
    assert result["metrics"]["model_tokens_to_first_useful_gain"] is None
    assert result["metrics"]["observer_wall_seconds"] is None
    assert result["metrics"]["observer_wall_seconds_known_lower_bound"] == 0
    assert result["usage"]["model_tokens"] == 57


def test_durable_observer_snapshot_cannot_mask_changed_candidate_source(tmp_path,monkeypatch):
    receipt=native_receipt(tmp_path)
    suite.write_json(tmp_path/"observer-metrics.json",METRICS)
    source=Path(receipt["candidate"]["source_path"])
    source.write_bytes(source.read_bytes()+b"\n// modified after freezing")
    forbid_new_native_activity(monkeypatch)
    with pytest.raises(ValueError,match="source hash mismatch"):
        suite.generation_adapter(entry("native_codex_vanilla"),object(),tmp_path,"existing-session",resume=True)


@pytest.mark.parametrize("status",["starting","failed"])
def test_observer_snapshot_does_not_authorize_repeating_an_incomplete_native_prompt(tmp_path,monkeypatch,status):
    native_receipt(tmp_path,status=status)
    suite.write_json(tmp_path/"observer-metrics.json",METRICS)
    forbid_new_native_activity(monkeypatch)
    with pytest.raises(RuntimeError,match="incomplete"):
        suite.generation_adapter(entry("native_codex_vanilla"),object(),tmp_path,"existing-session",resume=True)
