"""Storage admission and bounded dispatch with synthetic CPU journals only."""
from copy import deepcopy
from pathlib import PurePosixPath
import subprocess
import threading

import pytest

from optimizer_agent.codex_backend import CodexError
from optimizer_agent.execution_backend import BackendError, SlurmConfig
from optimizer_agent.suite_orchestrator import (SuiteOrchestrator, infrastructure_category,
    remote_storage_failure_directory, candidate_receipt, write_json)
from optimizer_agent.campaign import Candidate, LaunchRule
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from test_bootstrap_recovery import study


PQ = ('** WARNING: connection is not using a post-quantum key exchange algorithm.\n'
      '** This session may be vulnerable to "store now, decrypt later" attacks.\n'
      '** The server may need to be upgraded. See https://openssh.com/pq.html\n')
REMOTE = "/gpfs/scratch/cpu-fixture/optimizer/sessions/suite-gen-" + "a" * 32
REASON = "Authenticated worker staging mkdir failed because the remote filesystem quota was reached"


def diagnostic(path=REMOTE, reason="Disk quota exceeded", quotes=("‘", "’")):
    return f"mkdir: cannot create directory {quotes[0]}{path}{quotes[1]}: {reason}"


@pytest.mark.parametrize("reason", ["Disk quota exceeded", "No space left on device"])
@pytest.mark.parametrize("quotes", [("‘", "’"), ("'", "'"), ('"', '"')])
@pytest.mark.parametrize("banner", ["", PQ])
def test_exact_storage_diagnostic_is_identified_without_a_generic_io_guess(reason, quotes, banner):
    message = banner + diagnostic(reason=reason, quotes=quotes)
    assert remote_storage_failure_directory(message) == REMOTE
    assert infrastructure_category(BackendError(message)) == "remote_storage_failure"
    assert infrastructure_category(subprocess.CalledProcessError(1, ["ssh", "romeo"], stderr=message.encode())) == "remote_storage_failure"


@pytest.mark.parametrize("error", [
    "Disk quota exceeded", "io_error", "[Errno 122] Disk quota exceeded", "No space left on device",
    "compiler quoted:\n" + diagnostic(), "provider failed: " + diagnostic(),
    "Model response quoted:\n" + diagnostic(), "unexpected warning\n" + diagnostic(),
    PQ.splitlines()[0] + "\n" + diagnostic(), diagnostic() + "\nunknown trailing diagnostic",
    diagnostic("relative/sessions/worker"), diagnostic("/gpfs/scratch/cpu/optimizer/../sessions/worker"),
    diagnostic("/tmp/sessions/worker"), diagnostic("/gpfs/scratch/cpu/optimizer/not-sessions/worker"),
    diagnostic(reason="Permission denied"), diagnostic() + " (quoted example)",
    CodexError(PQ + diagnostic()),
    subprocess.CalledProcessError(1, ["nvcc", "kernel.cu"], stderr=diagnostic().encode()),
    subprocess.CalledProcessError(0, ["ssh", "romeo"], stderr=diagnostic().encode()),
])
def test_noise_provider_compiler_and_non_session_paths_are_not_storage_recovery(error):
    assert remote_storage_failure_directory(error) is None
    assert infrastructure_category(error) is None


def storage_failure(orchestrator, task, directory, *, path=None, saved_category=None):
    remote = str(PurePosixPath(SlurmConfig().remote_root) / "sessions" / task["session"])
    failure = {"error": PQ + diagnostic(path or remote), "job_id": None,
               "infrastructure_category": saved_category}
    write_json(directory / "failure.json", failure)
    orchestrator.journal.update(task["id"], result=failure, error=failure["error"])
    return failure


def test_storage_not_submitted_recovery_authenticates_old_owner_and_preserves_failure(tmp_path):
    value, task, directory, _, actions, _, _ = study(tmp_path, submitted=False, prefix=False)
    storage_failure(value, task, directory)
    before = (directory / "failure.json").read_bytes()
    audit = value.recover_bootstrap(task["id"], REASON)
    current = value.journal.task(task["id"])
    assert audit["infrastructure_category"] == "remote_storage_failure"
    assert audit["ownership_status"] == {"state": "NOT_SUBMITTED", "terminal": False}
    assert audit["phase"] == "committed" and audit["new_job_submitted"] is False
    assert audit["model_calls"] == audit["gpu_requests"] == 0
    assert current["state"] == "running" and current["job_id"] is None
    assert current["session"] == task["session"] and current["attempt"] == task["attempt"]
    assert (directory / "failure.json").read_bytes() == before
    assert all(action[0] in ("backend", "read") for action in actions)
    previous_actions = list(actions)
    with pytest.raises(ValueError, match="Only a failed, inactive generation"):
        value.recover_bootstrap(task["id"], REASON)
    assert actions == previous_actions


@pytest.mark.parametrize("damage", ["other_session", "forged_category", "submission_started", "submission_error",
    "local_ack", "hardware", "native_request", "model_intent", "failed_native_result"])
def test_storage_category_never_bypasses_path_model_or_submission_proofs(tmp_path, damage):
    value, task, directory, remote, actions, _, _ = study(tmp_path, submitted=False, prefix=False)
    failure = storage_failure(value, task, directory,
        path=REMOTE if damage == "other_session" else None,
        saved_category="remote_storage_failure" if damage == "forged_category" else None)
    if damage == "forged_category":
        failure["error"] = "quota word alone is not an authenticated diagnostic"
        write_json(directory / "failure.json", failure)
    elif damage.startswith("submission_"):
        remote["remote"][damage + ".json"] = {"attempted": True}
    elif damage == "local_ack":
        write_json(directory / "worker/slurm-session.json", {"job_id": "lost-ack"})
    elif damage == "hardware":
        write_json(directory / "worker/hardware.json", {})
    elif damage in ("native_request", "failed_native_result"):
        write_json(directory / "request-receipts/cpu-fixture.json",
                   {"report": {"ok": damage != "failed_native_result"}})
    elif damage == "model_intent":
        write_json(directory / "generation/model/cpu-intent.json", {})
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"], REASON)
    assert value.journal.task(task["id"])["state"] == "recovery_required"
    assert not (directory / "bootstrap-recovery.json").exists()
    if damage in ("other_session", "forged_category", "model_intent", "failed_native_result"):
        assert all(action[0] != "read" for action in actions)


def test_resuming_prepared_storage_recovery_rechecks_the_owned_directory(tmp_path, monkeypatch):
    value, task, directory, _, actions, _, _ = study(tmp_path, submitted=False, prefix=False)
    storage_failure(value, task, directory)
    original_update = value.journal.update
    def interrupt(key, **fields):
        if fields.get("state") == "running":
            raise OSError("synthetic preparation interruption")
        return original_update(key, **fields)
    monkeypatch.setattr(value.journal, "update", interrupt)
    with pytest.raises(OSError, match="synthetic preparation"):
        value.recover_bootstrap(task["id"], REASON)
    monkeypatch.setattr(value.journal, "update", original_update)
    factory = value.backend_factory
    def changed_directory(*args):
        backend = factory(*args)
        backend.remote_dir = REMOTE
        return backend
    value.backend_factory = changed_directory
    before = len(actions)
    with pytest.raises(ValueError, match="storage failure directory"):
        value.resume_bootstrap_preparation(task["id"])
    assert all(action[0] != "read" for action in actions[before:])
    assert value.journal.task(task["id"])["state"] == "recovery_required"


def queued_plan(count=83):
    plan = plan_suite(CampaignConfig(run_id="storage-queue", profile=False),
                      seeds=(1, 2, 3), include_greedy=True)
    plan["entries"] = plan["entries"][:count]
    return plan


def test_three_startup_failures_hold_79_tasks_and_final_while_active_worker_finishes(tmp_path):
    barrier = threading.Barrier(4)
    active_entered, release_active, failure_observed = (threading.Event() for _ in range(3))
    backend_calls, closed, returned, errors = [], [], [], []
    lock = threading.Lock()
    ids = sorted(entry["config"]["run_id"] for entry in queued_plan()["entries"])
    active_id = ids[3]
    class Backend:
        def __init__(self, task, config, directory):
            self.task, self.job_id = task, None
            with lock:
                backend_calls.append(task["id"])
        def start(self):
            barrier.wait(timeout=10)
            if self.task["id"] != active_id:
                raise BackendError(PQ + diagnostic())
            self.job_id = "900001"
        def wait_until_ready(self):
            pass
        def runner(self):
            return object()
        def close(self):
            closed.append(self.task["id"])
        def wait_until_finished(self, **kwargs):
            return {"terminal": True, "state": "COMPLETED", "exit_code": "0:0", "job_id": self.job_id}
    def generate(entry, runner, directory, session, **kwargs):
        assert entry["config"]["run_id"] == active_id
        active_entered.set()
        assert release_active.wait(timeout=10)
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "fixture.cu"
        source.write_text('extern "C" __global__ void fixture() {}')
        return {"generation_complete": True, "candidate": candidate_receipt(
            Candidate("cpu-fixture", source, "fixture", LaunchRule()))}
    def no_final(*args, **kwargs):
        pytest.fail("Storage failure must retain every future FINAL task")
    value = SuiteOrchestrator(queued_plan(), tmp_path, concurrency=4,
        backend_factory=Backend, generate=generate, qualify=no_final)
    original_status = value.status
    def observe_status():
        status = original_status()
        if any(row["state"] == "recovery_required" for row in status["tasks"]):
            failure_observed.set()
        return status
    value.status = observe_status
    def controller():
        try:
            returned.append(value.run())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=controller)
    thread.start()
    try:
        assert active_entered.wait(timeout=10) and failure_observed.wait(timeout=10)
        assert thread.is_alive() and closed == []
        assert set(backend_calls) == set(ids[:4])
        future_rows = [row for row in value.journal.tasks("generation") if row["id"] not in ids[:4]]
        assert len(future_rows) == 79
        assert all(row["state"] == "pending" and row["started"] is None and row["job_id"] is None for row in future_rows)
    finally:
        release_active.set()
        thread.join(timeout=15)
    assert not thread.is_alive() and errors == [] and len(returned) == 1
    assert closed == [active_id] and len(backend_calls) == 4
    rows = value.journal.tasks("generation")
    assert sum(row["state"] == "completed" for row in rows) == 1
    assert sum(row["state"] == "recovery_required" for row in rows) == 3
    assert sum(row["state"] == "pending" for row in rows) == 79
    assert all(row["state"] == "pending" for row in value.journal.tasks("final"))


def test_existing_unrecovered_storage_failure_holds_generation_and_final_without_retry(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("An unrecovered storage incident must not start another task")
    value = SuiteOrchestrator(queued_plan(4), tmp_path, backend_factory=forbidden)
    task = value.journal.tasks("generation")[0]
    value.journal.update(task["id"], state="recovery_required", result={"partial": True}, error=PQ + diagnostic())
    before = deepcopy(value.journal.tasks())
    value.run()
    assert value.journal.tasks() == before


def test_skipped_or_locked_snapshot_tasks_are_considered_once_without_requeue_loop(tmp_path):
    value = SuiteOrchestrator(queued_plan(9), tmp_path, backend_factory=lambda *args: None)
    calls = []
    value._task = lambda task: calls.append(task["id"])  # Simulate an independently held task lock.
    value.run()
    assert len(calls) == len(set(calls)) == 9
    assert all(row["state"] == "pending" and row["started"] is None for row in value.journal.tasks())
