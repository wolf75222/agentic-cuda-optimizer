"""Idempotent controller restart boundaries; no GPU, SSH or model activity."""
import json
from pathlib import Path
import time

import pytest

from optimizer_agent.campaign import Candidate,LaunchRule
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent import codex_backend,suite_orchestrator as suite,vanilla_gpu_service
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_report import write_report


def native_plan():
    plan=plan_suite(CampaignConfig(run_id="resume-boundary"),seeds=(1729,2718))
    plan["entries"]=[entry for entry in plan["entries"] if entry["mode"] == "native_codex_vanilla"
                     and entry["workload"] == "matmul" and entry["seed"] == 1729]
    plan["modes"]=["native_codex_vanilla"]
    plan["workloads"]=["matmul"]
    return plan


def no_activity(*args,**kwargs):
    pytest.fail("Historical/superseded study must not submit, launch a service, call a model or measure again")


def forbid_models(monkeypatch):
    monkeypatch.setattr(suite,"_script",no_activity)
    monkeypatch.setattr(codex_backend,"default_codex_executable",no_activity)
    monkeypatch.setattr(vanilla_gpu_service,"VanillaGPUService",no_activity)


def completed_cli(directory):
    protected=Path(directory)/"generation/evidence"
    protected.mkdir(parents=True)
    source=protected/"kernel.cu"
    source.write_bytes((suite.REPO/"examples/naive_matmul.cu").read_bytes())
    candidate=suite.candidate_receipt(Candidate("already-generated",source,"matmul",
        LaunchRule(kind="square_2d",block=(16,16,1))))
    receipt={"status":"completed","candidate":candidate,"started_at":1000.0,"ended_at":1010.0,
             "usage":{"input_tokens":30,"output_tokens":4},"usage_events":[{"turn_id":"completed-once"}]}
    suite.write_json(protected/"receipt.json",receipt)
    suite.write_json(protected/"candidate.json",candidate)
    suite.write_json(protected/"gain-observer/summary.json",{"identity":{"method":"native_codex"},
        "observations":{},"first_useful_gain":None,"metrics":{
            "first_useful_gain_status":"no_confirmed_gain","time_to_first_useful_gain_seconds":None,
            "model_tokens_to_first_useful_gain":None,"observer_wall_seconds":.25}})
    return candidate


class ReadonlyHistoricalBackend:
    def __init__(self,task,directory,calls,state,exit_code,finish_after_stop=False):
        self.task=task;self.directory=Path(directory);self.calls=calls
        self.observed_state=state;self.exit_code=exit_code
        self.finish_after_stop=finish_after_stop
        self.session_id=task["session"]
        self.remote_dir="/isolated-history/sessions/"+self.session_id
        self.job_id=None;self.historical_only=False
    def attach_existing(self):
        self.calls.append("attach_existing")
        receipt=json.loads((self.directory/"worker/slurm-session.json").read_text())
        assert receipt["session_id"] == self.session_id and receipt["job_id"] == "771234"
        assert receipt["remote_dir"] == self.remote_dir
        self.job_id=receipt["job_id"];self.historical_only=True
        return self.status()
    attach_readonly=attach_existing
    def start(self,allow_terminal=False):
        if not allow_terminal:
            no_activity()
        # Production API explicitly makes this flag an attach-only alias.
        self.calls.append("start_readonly_alias")
        return self.attach_existing()
    def stage(self):
        no_activity()
    def wait_until_ready(self):
        no_activity()
    def runner(self):
        no_activity()
    def require_live_worker(self,*args,**kwargs):
        no_activity()
    def status(self):
        assert self.historical_only,"Historical accounting must authenticate exact ownership first"
        self.calls.append("status")
        return {"job_id":self.job_id,"state":self.observed_state,"exit_code":self.exit_code,
                "terminal":self.observed_state in {"COMPLETED","FAILED","TIMEOUT"},
                "nodes":"simulated-historical-node","evidence_kind":"simulated_cpu_test"}
    def collect_diagnostics(self):
        self.calls.append("collect_diagnostics")
        return self.status()
    def wait_until_finished(self,timeout=60):
        self.calls.append("wait_until_finished_readonly")
        return {**self.status(),"terminal":self.observed_state in {"COMPLETED","FAILED","TIMEOUT"}}
    def close(self,cancel=False):
        # The controller may explicitly STOP its authenticated, still-running
        # worker after pure reconstruction. It must not scancel or STOP a job
        # that already has terminal accounting.
        assert not cancel and self.historical_only and self.observed_state == "RUNNING"
        self.calls.append("owned_graceful_STOP")
        if self.finish_after_stop:
            self.observed_state="COMPLETED";self.exit_code="0:0"


def historical_study(tmp_path,state="COMPLETED",exit_code="0:0",*,finish_after_stop=False):
    calls=[]
    def factory(task,config,directory):
        # Artifact/observer reconstruction must finish before even attaching
        # historical accounting, irrespective of the worker's current state.
        assert (Path(directory)/"result.json").is_file()
        assert (Path(directory)/"generation/observer-metrics.json").is_file()
        calls.append("backend_constructed")
        return ReadonlyHistoricalBackend(task,directory,calls,state,exit_code,finish_after_stop)
    orchestrator=suite.SuiteOrchestrator(native_plan(),tmp_path,backend_factory=factory)
    task=orchestrator.journal.tasks("generation")[0]
    directory=orchestrator._directory(task)
    candidate=completed_cli(directory)
    suite.write_json(directory/"worker/slurm-session.json",{
        "session_id":task["session"],"job_id":"771234",
        "remote_dir":"/isolated-history/sessions/"+task["session"]})
    orchestrator.journal.update(task["id"],state="running",job_id="771234",started=time.time()-3)
    task=orchestrator.journal.tasks("generation")[0]
    assert not (directory/"result.json").exists()
    assert not (directory/"generation/observer-metrics.json").exists()
    return orchestrator,task,directory,candidate,calls


def test_superseded_study_blocks_run_and_recovery_before_backend_or_model_but_allows_reports(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator=suite.SuiteOrchestrator(native_plan(),tmp_path,backend_factory=no_activity,
        generate=no_activity,qualify=no_activity)
    final=orchestrator.journal.tasks("final")[0]
    orchestrator.journal.update(final["id"],state="recovery_required",job_id="771234",
        result={"infrastructure_category":"transport_timeout"})
    suite.write_json(tmp_path/"superseded-by.json",{"directory":"/new-study","reason":"New immutable isolation protocol"})
    with pytest.raises((RuntimeError,ValueError),match="[Ss]uperseded"):
        orchestrator.run()
    with pytest.raises((RuntimeError,ValueError),match="[Ss]uperseded"):
        orchestrator.recover_final(final["id"],"Recorded transport failure in the old superseded study")
    status=orchestrator.status()
    report=write_report(tmp_path/"read-only-report",status)
    assert report["summary"]["complete"] is False
    assert (tmp_path/"read-only-report/report.json").is_file()
    assert orchestrator.journal.tasks("generation")[0]["state"] == "pending"


def test_completed_cli_gap_reconstructs_generation_and_observer_without_new_worker_or_prompt(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator,task,directory,candidate,calls=historical_study(tmp_path)
    before=(directory/"generation/evidence/receipt.json").read_bytes()
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    assert row["state"] == "completed" and row["job_id"] == "771234"
    result=row["result"]
    assert result["candidate"]["artifact_hash"] == candidate["artifact_hash"]
    assert result["generation_complete"] and result["execution_complete"]
    assert result["metrics"]["observer_wall_seconds"] == .25
    assert result["metrics"]["time_to_first_useful_gain_seconds"] is None
    assert result["worker_terminal"]["state"] == "COMPLETED"
    assert result["session_id"] == task["session"] and result["job_id"] == "771234"
    assert "attach_existing" in calls and calls.count("backend_constructed") == 1
    assert (directory/"generation/observer-metrics.json").is_file()
    assert (directory/"generation/evidence/receipt.json").read_bytes() == before
    assert (directory/"result.json").is_file()
    # Repeating the controller boundary reuses its authenticated durable result.
    count=len(calls)
    orchestrator._task(row)
    assert len(calls) == count
    assert orchestrator.journal.tasks("generation")[0]["result"]["candidate"]["artifact_hash"] == candidate["artifact_hash"]


@pytest.mark.parametrize("state,exit_code",[("FAILED","1:0"),("TIMEOUT","0:15")])
def test_historical_failed_worker_is_preserved_without_relabel_or_stop(tmp_path,monkeypatch,state,exit_code):
    forbid_models(monkeypatch)
    orchestrator,task,_,candidate,calls=historical_study(tmp_path,state,exit_code)
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    assert row["state"] != "completed"
    result=row["result"]
    assert result["candidate"]["artifact_hash"] == candidate["artifact_hash"]
    assert result["generation_complete"] is True and result["execution_complete"] is False
    assert result["worker_terminal"]["state"] == state and result["worker_terminal"]["terminal"] is True
    assert result["worker_terminal"]["exit_code"] == exit_code and result["job_id"] == "771234"
    assert "attach_existing" in calls
    assert "owned_graceful_STOP" not in calls


def test_reconstructed_native_gap_can_gracefully_close_its_exact_running_worker(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator,task,_,candidate,calls=historical_study(tmp_path,"RUNNING",None,finish_after_stop=True)
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    result=row["result"]
    assert row["state"] == "completed" and result["execution_complete"] is True
    assert result["candidate"]["artifact_hash"] == candidate["artifact_hash"]
    assert result["worker_terminal"]["state"] == "COMPLETED"
    assert result["worker_terminal"]["terminal"] is True and result["worker_terminal"]["exit_code"] == "0:0"
    assert result["session_id"] == task["session"] and result["job_id"] == "771234"
    assert calls.count("owned_graceful_STOP") == 1
    assert calls.index("attach_existing") < calls.index("owned_graceful_STOP") < calls.index("wait_until_finished_readonly")


def test_running_worker_after_graceful_stop_remains_incomplete(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator,task,_,candidate,calls=historical_study(tmp_path,"RUNNING",None)
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    result=row["result"]
    assert row["state"] == "recovery_required"
    assert result["candidate"]["artifact_hash"] == candidate["artifact_hash"]
    assert result["generation_complete"] is True and result["execution_complete"] is False
    assert result["worker_terminal"]["state"] == "RUNNING"
    assert result["worker_terminal"]["terminal"] is False and result["worker_terminal"]["exit_code"] is None
    assert result["session_id"] == task["session"] and result["job_id"] == "771234"
    assert result["infrastructure_category"] == "worker_terminal_incomplete"
    assert calls.count("owned_graceful_STOP") == 1


def test_cli_receipt_does_not_authenticate_a_changed_source_during_gap_recovery(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator,task,_,candidate,calls=historical_study(tmp_path)
    source=Path(candidate["source_path"])
    source.write_bytes(source.read_bytes()+b"\n// unexpected source replacement")
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    assert row["state"] == "recovery_required"
    assert "source hash mismatch" in row["error"]
    assert calls == []


def test_native_gap_cannot_stop_a_worker_different_from_the_owned_journal_job(tmp_path,monkeypatch):
    forbid_models(monkeypatch)
    orchestrator,task,directory,_,calls=historical_study(tmp_path,"RUNNING",None)
    orchestrator.journal.update(task["id"],job_id="771999")
    task=orchestrator.journal.tasks("generation")[0]
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    failure=json.loads((directory/"failure.json").read_text())
    assert row["state"] == "recovery_required"
    assert "differs from the journal's owned job" in row["error"]
    assert failure["job_id"] is None and failure["execution_complete"] is False
    assert "attach_existing" in calls and "owned_graceful_STOP" not in calls
    assert "wait_until_finished_readonly" not in calls


def test_fresh_native_read_isolation_failure_blocks_before_backend_construction(tmp_path,monkeypatch):
    from optimizer_agent import native_read_isolation
    calls=[]
    def failed_preflight(sensitive_root,*,codex_executable):
        calls.append((sensitive_root,codex_executable))
        return {"ok":False,"evidence_kind":"simulated_cpu_test","reason":"Protected study data remained readable"}
    forbid_models(monkeypatch)
    monkeypatch.setattr(codex_backend,"default_codex_executable",lambda:Path("/never-executed/codex"))
    monkeypatch.setattr(native_read_isolation,"probe_read_isolation",failed_preflight)
    orchestrator=suite.SuiteOrchestrator(native_plan(),tmp_path,backend_factory=no_activity)
    task=orchestrator.journal.tasks("generation")[0]
    orchestrator._task(task)
    row=orchestrator.journal.tasks("generation")[0]
    assert row["state"] == "recovery_required" and row["job_id"] is None
    assert "read-isolation preflight did not pass" in row["error"]
    assert calls == [(tmp_path.parent,Path("/never-executed/codex"))]
    receipt=json.loads((orchestrator._directory(task)/"native-read-isolation-preflight.json").read_text())
    assert receipt["ok"] is False and receipt["evidence_kind"] == "simulated_cpu_test"
