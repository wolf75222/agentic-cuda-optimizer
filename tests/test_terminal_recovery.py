"""CPU-only terminal transaction fixtures, never a real worker/model/study."""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from optimizer_agent import terminal_recovery as recovery
from optimizer_agent import suite_orchestrator as orchestration
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate, LaunchRule
from optimizer_agent.execution_backend import SlurmConfig
from optimizer_agent.suite_orchestrator import SuiteOrchestrator, candidate_receipt, write_json


class Crash(BaseException):
    pass


def forbidden(*args, **kwargs):
    pytest.fail("Terminal metadata recovery must never execute/stop GPU/model work")


@pytest.fixture
def completed(tmp_path):
    config = CampaignConfig(run_id="terminal-unit", search_sizes=(17,), profile=False)
    entry = {"config": config.to_dict(), "mode": "full", "group_id": "unit-group", "workload": "matmul"}
    actions, worker = [], {"proof": {"job_id": "101", "state": "COMPLETED", "exit_code": "0:0",
        "terminal": True, "elapsed": "00:01:07", "nodes": "fixture-node", "source": "fixture_scheduler"}}

    class Backend:
        def __init__(self, task, directory):
            self.config = SlurmConfig()
            self.results_dir = directory / "worker"
            self.session_id, self.job_id = task["session"], None
            self.remote_dir = self.config.remote_root + "/sessions/" + self.session_id
            self.historical_only = False

        def attach_existing(self):
            actions.append("attach_existing")
            self.historical_only = True
            self.job_id = worker["proof"]["job_id"]
            return deepcopy(worker["proof"])

        start = close = runner = wait_until_ready = wait_until_finished = forbidden

    suite = SuiteOrchestrator({"entries": [entry], "modes": ["full"]}, tmp_path / "study",
        backend_factory=lambda task, config, directory: Backend(task, directory), generate=forbidden, qualify=forbidden)
    task = suite.journal.tasks("generation")[0]
    directory = suite._directory(task)
    generation = directory / "generation"
    generation.mkdir(parents=True)
    source = generation / "best.cu"
    source.write_text('extern "C" __global__ void matmul(const float*a,const float*b,float*c,int n){}')
    frozen = json.loads(json.dumps(candidate_receipt(Candidate(
        "retained", source, "matmul", LaunchRule(block=(128, 1, 1))))))
    result = {"run_id": task["id"], "session_id": task["session"], "job_id": "101",
        "task_evidence_path": str(directory), "status": "generation_complete", "method_status": "completed",
        "generation_complete": True, "generation_only": True, "trials_done": 3,
        "contract_hash": "c" * 64, "best_candidate_id": frozen["candidate_id"], "artifact_hash": frozen["artifact_hash"],
        "candidate": frozen, "metrics": {"correct_candidates": 3, "research_seconds": 7.5},
        "usage": {"model_calls": 5, "model_tokens": None}, "final": {"status": "deferred"}}
    write_json(directory / "result.json", result)
    failure = {"status": "failed", "session_id": task["session"], "job_id": "101", "execution_complete": False,
        "error": "STOP upload: [Errno 122] Disk quota exceeded", "partial_result": deepcopy(result)}
    write_json(directory / "failure.json", failure)
    write_json(generation / "config.json", config.to_dict())
    write_json(generation / "contract.json", {"contract_hash": "c" * 64, "contract": {"unit": True}})
    backend = Backend(task, directory)
    write_json(directory / "worker/slurm-session.json", {"session_id": task["session"], "job_id": "101",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    write_json(directory / "worker/slurm-diagnostics/accounting.json", {"job_id": "101", "state": "RUNNING"})
    write_json(generation / "model" / ("a" * 32 + ".json"), {"call_id": "a" * 32, "status": "completed",
        "returncode": 0, "cli_launch": {"state": "exited"}, "usage": {"input_tokens": None}})
    suite.journal.update(task["id"], state="recovery_required", job_id="101", result=result,
        error=failure["error"], started=100., finished=200.)
    task = suite.journal.task(task["id"])
    return suite, task, directory, result, worker, actions


def reconcile(value):
    suite, task, *_ = value
    return recovery.reconcile_completed_generation(suite, task["id"],
        "Reconcile the already completed fixture generation after its STOP upload failure")


def folder(value):
    suite, task, *_ = value
    return suite.directory / "terminal-reconciliations" / hashlib.sha256(task["id"].encode()).hexdigest() / "000"


def update(path, **fields):
    value = json.loads(path.read_bytes())
    value.update(fields)
    write_json(path, value)


def test_completed_generation_uses_existing_path_without_gpu_model_or_stop(completed):
    suite, task, directory, original, worker, actions = completed
    result_raw = (directory / "result.json").read_bytes()
    failure_raw = (directory / "failure.json").read_bytes()
    previous_accounting = (directory / "worker/slurm-diagnostics/accounting.json").read_bytes()
    commit = reconcile(completed)
    after = suite.journal.task(task["id"])
    current = json.loads((directory / "result.json").read_bytes())
    assert after["state"] == "completed" and after["attempt"] == task["attempt"] and after["session"] == task["session"]
    assert after["error"] == task["error"] and after["result"] == current
    assert recovery._science(current) == original
    assert current["worker_terminal"] == worker["proof"] and current["execution_complete"] is True
    assert current["usage"]["model_tokens"] is None
    assert (folder(completed) / "result.json.original").read_bytes() == result_raw
    assert (folder(completed) / "failure.json.original").read_bytes() == failure_raw
    assert (directory / "failure.json").read_bytes() == failure_raw
    assert (folder(completed) / "accounting.original.json").read_bytes() == previous_accounting
    accounting = directory / "worker/slurm-diagnostics/accounting.json"
    assert json.loads(accounting.read_bytes()) == worker["proof"]
    assert commit["accounting_sha256"] == hashlib.sha256(accounting.read_bytes()).hexdigest()
    assert commit["gpu_requests_during_reconciliation"] == commit["model_calls_during_reconciliation"] == 0
    assert actions == ["attach_existing"]


def test_committed_replay_is_local_readonly_and_preserves_ledger_unknown(completed):
    suite, task, directory, original, _, actions = completed
    commit = reconcile(completed)
    before = {p: p.read_bytes() for p in (directory / "result.json", directory / "failure.json",
        directory / "worker/slurm-diagnostics/accounting.json", folder(completed) / "prepared.json", folder(completed) / "committed.json")}
    snapshot = suite.journal.task(task["id"])
    suite.backend_factory = forbidden
    assert reconcile(completed) == commit
    assert suite.journal.task(task["id"]) == snapshot
    assert all(p.read_bytes() == b for p, b in before.items())
    assert actions == ["attach_existing"] and snapshot["result"]["usage"] == original["usage"]


@pytest.mark.parametrize("change", ["pending", "method_failed", "partial", "source", "candidate", "ack_session",
    "ack_config", "ack_job", "contract", "config", "failure_identity", "failure_partial", "exposed", "cli_active"])
def test_invalid_completed_context_refuses_before_preparation_or_journal_mutation(completed, change):
    suite, task, directory, *_ = completed
    if change == "pending":
        suite.journal.update(task["id"], state="pending")
    elif change in {"method_failed", "partial"}:
        fields = {"method_status": "failed"} if change == "method_failed" else {"generation_complete": False}
        update(directory / "result.json", **fields)
    elif change == "source":
        (directory / "generation/best.cu").write_text("changed")
    elif change == "candidate":
        c = deepcopy(completed[3]["candidate"]);c["parameters"] = {"TILE": 9}
        update(directory / "result.json", candidate=c)
    elif change.startswith("ack_"):
        field, value = {"ack_session": ("session_id", "other"), "ack_job": ("job_id", "102"),
                        "ack_config": ("config", {})}[change]
        update(directory / "worker/slurm-session.json", **{field: value})
    elif change == "contract":
        update(directory / "generation/contract.json", contract_hash="d" * 64)
    elif change == "config":
        update(directory / "generation/config.json", max_trials=99)
    elif change == "failure_identity":
        update(directory / "failure.json", session_id="other")
    elif change == "failure_partial":
        update(directory / "failure.json", partial_result={"status": "failed"})
    elif change == "exposed":
        write_json(directory / "generation/holdout-exposed.json", {"exposed": True})
    elif change == "cli_active":
        update(directory / "generation/model" / ("a" * 32 + ".json"), status="running", returncode=None)
    before = suite.journal.task(task["id"])
    originals = {p: p.read_bytes() for p in (directory / "result.json", directory / "failure.json")}
    with pytest.raises(ValueError):
        reconcile(completed)
    assert not folder(completed).exists() and suite.journal.task(task["id"]) == before
    assert all(p.read_bytes() == b for p, b in originals.items())


@pytest.mark.parametrize("proof", [
    {"state": "RUNNING", "terminal": False}, {"state": "ACCOUNTING_PENDING", "terminal": False},
    {"state": "FAILED", "terminal": True}, {"exit_code": "1:0"}, {"job_id": "102"}])
def test_non_successful_or_mismatched_worker_proof_never_mutates(completed, proof):
    suite, task, directory, _, worker, _ = completed
    worker["proof"].update(proof)
    before = suite.journal.task(task["id"])
    with pytest.raises(ValueError):
        reconcile(completed)
    assert not folder(completed).exists() and suite.journal.task(task["id"]) == before


def test_final_payload_list_and_active_locks_refuse_without_backend(completed):
    suite, task, *_ = completed
    suite.backend_factory = forbidden
    final = suite.journal.tasks("final")[0]
    with pytest.raises(ValueError, match="Only generation"):
        recovery.reconcile_completed_generation(suite, final["id"], "Final tasks are outside this recovery protocol")
    lock = suite.directory / "task-locks" / (hashlib.sha256(task["id"].encode()).hexdigest() + ".lock")
    lock.parent.mkdir(exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still active"):
            reconcile(completed)
    with suite.journal.controller(), pytest.raises(RuntimeError, match="Another suite controller"):
        reconcile(completed)
    assert not folder(completed).exists()


@pytest.mark.parametrize("phase", ["backup_result", "before_prepared", "prepared", "before_accounting",
    "after_accounting", "before_terminal_commit", "result", "journal", "before_commit", "after_commit"])
def test_each_prepared_crash_window_resumes_same_transaction_without_execution(completed, monkeypatch, phase):
    suite, task, directory, original, _, actions = completed
    real_write, real_terminal = recovery._write, SuiteOrchestrator._commit_terminal_result
    real_update, real_json = suite.journal.update, orchestration.write_json
    real_raw, real_accounting = recovery._publish_raw, recovery._install_accounting
    def publish_raw(path, value, root):
        real_raw(path, value, root)
        if phase == "backup_result" and Path(path).name == "result.json.original":raise Crash()
    def install_accounting(*args):
        if phase == "before_accounting":raise Crash()
        real_accounting(*args)
        if phase == "after_accounting":raise Crash()
    def write(path, value, root):
        if Path(path).name == "prepared.json" and phase == "before_prepared":
            raise Crash()
        if Path(path).name == "committed.json" and phase == "before_commit":
            raise Crash()
        real_write(path, value, root)
        if Path(path).name == ("prepared.json" if phase == "prepared" else "committed.json" if phase == "after_commit" else "never"):
            raise Crash()
    def terminal_commit(self, *args, **kwargs):
        if phase == "before_terminal_commit":raise Crash()
        return real_terminal(self, *args, **kwargs)
    def update_journal(key, **values):
        real_update(key, **values)
        if phase == "journal" and values.get("state") == "completed":
            raise Crash()
    def write_json(path, value):
        real_json(path, value)
        if phase == "result" and Path(path) == directory / "result.json" and value.get("execution_complete") is True:
            raise Crash()
    with monkeypatch.context() as m:
        m.setattr(recovery, "_write", write);m.setattr(SuiteOrchestrator, "_commit_terminal_result", terminal_commit)
        m.setattr(suite.journal, "update", update_journal);m.setattr(orchestration, "write_json", write_json)
        m.setattr(recovery, "_publish_raw", publish_raw);m.setattr(recovery, "_install_accounting", install_accounting)
        with pytest.raises(Crash):reconcile(completed)
    prepared_path = folder(completed) / "prepared.json"
    prepared = prepared_path.read_bytes() if prepared_path.exists() else None
    if phase in {"journal", "before_commit", "after_commit"}:suite.backend_factory = forbidden
    reconcile(completed)
    assert suite.journal.task(task["id"])["state"] == "completed"
    if prepared is not None:assert prepared_path.read_bytes() == prepared
    assert recovery._science(json.loads((directory / "result.json").read_bytes())) == original
    assert len(actions) == (1 if phase in {"journal", "before_commit", "after_commit"} else 2)


@pytest.mark.parametrize("tamper", ["metrics", "failure", "source", "ack", "backup", "accounting"])
def test_prepared_reconciliation_refuses_late_evidence_changes(completed, monkeypatch, tamper):
    suite, task, directory, *_ = completed
    with monkeypatch.context() as m:
        m.setattr(SuiteOrchestrator, "_commit_terminal_result", lambda *args, **kwargs: (_ for _ in ()).throw(Crash()))
        with pytest.raises(Crash):reconcile(completed)
    if tamper == "metrics":update(directory / "result.json", metrics={"incorrect": 1})
    elif tamper == "failure":update(directory / "failure.json", error="changed")
    elif tamper == "source":(directory / "generation/best.cu").write_text("changed")
    elif tamper == "ack":update(directory / "worker/slurm-session.json", job_id="102")
    elif tamper == "backup":(folder(completed) / "result.json.original").write_text("changed")
    elif tamper == "accounting":update(directory / "worker/slurm-diagnostics/accounting.json", state="FAILED")
    before = suite.journal.task(task["id"])
    with pytest.raises(ValueError):reconcile(completed)
    assert suite.journal.task(task["id"]) == before and not (folder(completed) / "committed.json").exists()


def test_no_previous_accounting_is_distinct_from_a_fabricated_zero(completed):
    suite, task, directory, *_ = completed
    (directory / "worker/slurm-diagnostics/accounting.json").unlink()
    reconcile(completed)
    prepared = json.loads((folder(completed) / "prepared.json").read_bytes())
    assert prepared["accounting"]["previous_sha256"] is None
    assert not (folder(completed) / "accounting.original.json").exists()
    assert json.loads((directory / "worker/slurm-diagnostics/accounting.json").read_bytes())["elapsed"] == "00:01:07"


def test_tampered_commit_does_not_gain_authority(completed):
    reconcile(completed)
    update(folder(completed) / "committed.json", new_job_submitted=True)
    with pytest.raises(ValueError, match="Committed"):
        reconcile(completed)


@pytest.mark.parametrize("phase", ["before_result", "after_result", "before_journal", "after_journal"])
def test_ordinary_ioerror_never_overwrites_failure_and_prepared_remains_resumable(completed, monkeypatch, phase):
    suite, task, directory, original, *_ = completed
    failure = (directory / "failure.json").read_bytes()
    real_json, real_update = orchestration.write_json, suite.journal.update
    def write_json(path, value):
        target = Path(path) == directory / "result.json"
        if target and phase == "before_result":raise OSError("fixture result publication failed")
        real_json(path, value)
        if target and phase == "after_result":raise OSError("fixture after result publication")
    def update_journal(key, **values):
        target = values.get("state") == "completed"
        if target and phase == "before_journal":raise OSError("fixture journal commit failed")
        real_update(key, **values)
        if target and phase == "after_journal":raise OSError("fixture after journal commit")
    with monkeypatch.context() as m:
        m.setattr(orchestration, "write_json", write_json);m.setattr(suite.journal, "update", update_journal)
        with pytest.raises(OSError, match="fixture"):
            reconcile(completed)
    assert (directory / "failure.json").read_bytes() == failure
    assert (folder(completed) / "failure.json.original").read_bytes() == failure
    assert (folder(completed) / "prepared.json").is_file() and not (folder(completed) / "committed.json").exists()
    if phase == "after_journal":suite.backend_factory = forbidden
    reconcile(completed)
    assert (directory / "failure.json").read_bytes() == failure
    assert suite.journal.task(task["id"])["state"] == "completed"
    assert recovery._science(json.loads((directory / "result.json").read_bytes())) == original


def test_prepared_retry_finishes_accounting_directory_fsync_after_replace(completed, monkeypatch):
    from optimizer_agent import dispatch_attestation
    suite, task, directory, *_ = completed
    parent = directory / "worker/slurm-diagnostics"
    real_sync, accounting_syncs = dispatch_attestation._fsync_directory, []
    def sync(path):
        if Path(path) == parent:
            accounting_syncs.append(path)
            if len(accounting_syncs) == 1:raise OSError("fixture directory fsync interruption")
        real_sync(path)
    monkeypatch.setattr(dispatch_attestation, "_fsync_directory", sync)
    with pytest.raises(OSError, match="fixture directory fsync"):
        reconcile(completed)
    raw = (parent / "accounting.json").read_bytes()
    assert (folder(completed) / "prepared.json").exists()
    assert suite.journal.task(task["id"])["state"] == "recovery_required"
    reconcile(completed)
    assert len(accounting_syncs) == 2 and (parent / "accounting.json").read_bytes() == raw
    assert suite.journal.task(task["id"])["state"] == "completed"


@pytest.mark.parametrize("role", ["source", "accounting"])
def test_aliases_are_refused_without_reading_their_payload(completed, tmp_path, monkeypatch, role):
    suite, task, directory, *_ = completed
    victim = tmp_path / "do-not-read.json"
    victim.write_text("private fixture sentinel")
    path = directory / ("generation/best.cu" if role == "source" else "worker/slurm-diagnostics/accounting.json")
    path.unlink();path.symlink_to(victim)
    real_read = recovery._raw
    def read(value, root):
        assert Path(value) != victim
        return real_read(value, root)
    monkeypatch.setattr(recovery, "_raw", read)
    with pytest.raises(ValueError, match="Aliased"):reconcile(completed)
    assert not folder(completed).exists() and suite.journal.task(task["id"])["state"] == "recovery_required"
