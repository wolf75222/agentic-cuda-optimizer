"""CPU-only dispatch metadata tests; no Slurm, SSH, GPU or model execution."""
from concurrent.futures import Future
import hashlib
import json
import math

import pytest

from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import SuiteOrchestrator, _hash, write_json


@pytest.fixture
def suite(tmp_path):
    plan = plan_suite(CampaignConfig(run_id="dispatch-unit", search_sizes=(64,), profile=False), seeds=(1, 2))
    plan["entries"] = [entry for entry in plan["entries"] if entry["workload"] == "matmul" and entry["seed"] == 1]
    def forbidden_backend(*args):
        pytest.fail("A metadata dispatch test must not create a backend")
    return SuiteOrchestrator(plan, tmp_path, backend_factory=forbidden_backend)


def update(suite, task, **values):
    suite.journal.update(task["id"], **values)
    return suite.journal.task(task["id"])


def replay_proof(suite, task, *, started=10):
    task = update(suite, task, state="running", job_id="772379", started=started)
    directory = suite._directory(task)
    hardware = {"slurm_job_id": task["job_id"], "gpu_uuids": ["CPU-TEST-NOT-A-REAL-GPU"]}
    # Only metadata is needed for a priority hint. The actual replay validator
    # still rejects this deliberately incomplete receipt before generation.
    replay = {"schema_version": 1, "kind": "pre_model_bootstrap_prefix",
              "receipt_directory": str(directory / "request-receipts"),
              "hardware": hardware, "hardware_hash": _hash(hardware),
              "entries": [{"position": 0, "kind": "completed"}]}
    replay["manifest_hash"] = _hash(replay)
    replay_path = directory / "bootstrap-replay-manifest.json"
    write_json(replay_path, replay)
    audit = {"schema_version": 1, "phase": "committed", "task_id": task["id"],
             "previous_session": task["session"], "previous_job_id": task["job_id"],
             "same_live_allocation": True, "created_at_unix": 2,
             "replay_manifest": str(replay_path),
             "replay_manifest_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest()}
    ack = {"session_id": task["session"], "job_id": task["job_id"],
           "config": {"remote_root": "/cpu-test-owned-root"},
           "remote_dir": "/cpu-test-owned-root/sessions/" + task["session"]}
    write_json(directory / "bootstrap-recovery.json", audit)
    write_json(directory / "worker" / "slurm-session.json", ack)
    return task, audit, replay, ack


def unscored_proof(suite, task, *, created=20):
    task = update(suite, task, attempt=1, session="cpu-test-new-unscored-session")
    generation = suite._directory(task) / "generation"
    marker = generation / "unscored-rehost.json"
    source_recovery = {"kind": "unscored_performance_context_transfer", "context_id": "cpu-test-context",
                       "fresh_research_required": True, "marker_path": str(marker)}
    with suite.journal.connection() as db:
        db.execute("UPDATE tasks SET payload=? WHERE id=?",
                   (json.dumps({**task["payload"], "source_recovery": source_recovery}), task["id"]))
    task = suite.journal.task(task["id"])
    intent = {"schema_version": 1, "phase": "committed", "kind": source_recovery["kind"],
              "task_id": task["id"], "context_id": source_recovery["context_id"],
              "new_session": task["session"], "new_attempt": task["attempt"],
              "marker_path": str(marker), "new_generation": str(generation), "created_at_unix": created}
    write_json(marker, intent)
    write_json(suite.directory / "unscored-rehost-intents" / (task["id"] + ".json"), intent)
    return task, intent


def test_owned_allocation_replay_precedes_unscored_running_and_fresh(suite):
    tasks = suite.journal.tasks("generation")
    fresh, running, recovered, unscored, replay = tasks[:5]
    running = update(suite, running, state="running", started=1)
    recovered = update(suite, recovered, attempt=1, session="cpu-test-new-terminal-session")
    write_json(suite._directory({**recovered, "attempt": 0}) / "bootstrap-recovery.json",
               {"schema_version": 1, "phase": "committed", "task_id": recovered["id"],
                "same_live_allocation": False, "new_session": recovered["session"],
                "new_attempt": 1, "created_at_unix": 3})
    unscored, _ = unscored_proof(suite, unscored)
    replay, _, _, _ = replay_proof(suite, replay)
    expected = [replay["id"], unscored["id"], running["id"], recovered["id"], fresh["id"]]
    for rows in ([fresh, running, recovered, unscored, replay], [replay, recovered, fresh, unscored, running]):
        assert [row["id"] for row in sorted(rows, key=suite._dispatch_key)] == expected
    assert suite._dispatch_key(replay) == (0, 10.0, replay["id"], 0)
    assert suite._dispatch_key(unscored) == (1, 20.0, unscored["id"], 1)


@pytest.mark.parametrize("change", ["audit_task", "audit_phase", "audit_session", "audit_job", "audit_path", "audit_sha",
                                          "ack_session", "ack_job", "ack_remote_dir", "ack_uncertain", "ack_job_ids",
                                          "manifest_schema", "manifest_hardware", "manifest_empty"])
def test_ambiguous_replay_metadata_loses_only_extra_priority(suite, change):
    task, audit, replay, ack = replay_proof(suite, suite.journal.tasks("generation")[0])
    directory = suite._directory(task)
    if change.startswith("audit_"):
        key, value = {"audit_task": ("task_id", "different"), "audit_phase": ("phase", "prepared"),
                      "audit_session": ("previous_session", "different"), "audit_job": ("previous_job_id", "772380"),
                      "audit_path": ("replay_manifest", "/different"), "audit_sha": ("replay_manifest_sha256", "bad")}[change]
        audit[key] = value
    elif change.startswith("ack_"):
        key, value = {"ack_session": ("session_id", "different"), "ack_job": ("job_id", "772380"),
                      "ack_remote_dir": ("remote_dir", "/different"), "ack_uncertain": ("uncertain", True),
                      "ack_job_ids": ("job_ids", ["772379", "772380"])}[change]
        ack[key] = value
    else:
        if change == "manifest_schema":
            replay["schema_version"] = 2
        elif change == "manifest_hardware":
            replay["hardware"]["slurm_job_id"] = "772380"
            replay["hardware_hash"] = _hash(replay["hardware"])
        else:
            replay["entries"] = []
        replay["manifest_hash"] = _hash({k:v for k,v in replay.items() if k != "manifest_hash"})
        write_json(directory / "bootstrap-replay-manifest.json", replay)
        audit["replay_manifest_sha256"] = hashlib.sha256((directory / "bootstrap-replay-manifest.json").read_bytes()).hexdigest()
    write_json(directory / "bootstrap-recovery.json", audit)
    write_json(directory / "worker" / "slurm-session.json", ack)
    assert suite._dispatch_key(task) == (2, 10.0, task["id"], 0)
    assert suite.journal.task(task["id"])["state"] == "running"


@pytest.mark.parametrize("change", ["bytes", "context", "session", "attempt", "phase", "path"])
def test_unscored_priority_requires_committed_exact_marker_bytes(suite, change):
    task, intent = unscored_proof(suite, suite.journal.tasks("generation")[0])
    path = suite.directory / "unscored-rehost-intents" / (task["id"] + ".json")
    if change == "bytes":
        path.write_text(path.read_text() + "\n")
    else:
        key, value = {"context": ("context_id", "different"), "session": ("new_session", "different"),
                      "attempt": ("new_attempt", 2), "phase": ("phase", "prepared"),
                      "path": ("marker_path", "/different")}[change]
        intent[key] = value
        write_json(path, intent)
        write_json(suite._directory(task) / "generation" / "unscored-rehost.json", intent)
    assert suite._dispatch_key(task) == (3, math.inf, task["id"], 1)


def test_missing_invalid_timestamp_never_gets_fictitious_zero(suite):
    tasks = suite.journal.tasks("generation")[:3]
    tasks = [update(suite, task, state="running") for task in tasks]
    tasks[0]["started"] = float("nan")
    tasks[1]["started"] = True
    tasks[2]["started"] = None
    assert [row["id"] for row in sorted(reversed(tasks), key=suite._dispatch_key)] == sorted(row["id"] for row in tasks)
    assert all(suite._dispatch_key(row)[1] == math.inf for row in tasks)


def test_run_submits_in_priority_order_with_cpu_fake_executor(suite, monkeypatch):
    tasks = suite.journal.tasks("generation")
    replay, _, _, _ = replay_proof(suite, tasks[-1])
    unscored, _ = unscored_proof(suite, tasks[-2])
    submitted = []
    class FakeExecutor:
        def __init__(self, **kwargs):
            assert kwargs == {"max_workers": suite.concurrency}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, task):
            submitted.append(task["id"])
            future = Future()
            future.set_result(None)
            return future
    monkeypatch.setattr("optimizer_agent.suite_orchestrator.ThreadPoolExecutor", FakeExecutor)
    suite.run()
    assert submitted[:2] == [replay["id"], unscored["id"]]
    assert submitted[2:] == sorted(row["id"] for row in tasks[:-2])
    assert len(submitted) == len(tasks)  # No final group before its generators finish.


@pytest.mark.parametrize("change", ["completed", "failed", "recovery_required", "session", "attempt", "payload"])
def test_stale_queued_identity_or_terminal_state_never_executes(suite, monkeypatch, change):
    task = update(suite, suite.journal.tasks("generation")[0], state="running")
    if change in {"completed", "failed", "recovery_required"}:
        update(suite, task, state=change)
    elif change == "payload":
        with suite.journal.connection() as db:
            db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps({**task["payload"], "changed": True}), task["id"]))
    else:
        update(suite, task, **{change: "different-session" if change == "session" else 1})
    monkeypatch.setattr(suite, "_execute_task", lambda row: pytest.fail("Stale queued work executed"))
    suite._task(task)


def test_same_identity_uses_fresh_job_and_state_after_lock(suite, monkeypatch):
    task = suite.journal.tasks("generation")[0]
    fresh = update(suite, task, state="running", job_id="772379", started=15)
    executed = []
    monkeypatch.setattr(suite, "_execute_task", executed.append)
    suite._task(task)
    assert executed == [fresh]


@pytest.mark.parametrize("prepared_kind", ["bootstrap", "unscored"])
def test_priority_never_bypasses_incomplete_preparation_guard(suite, monkeypatch, prepared_kind):
    task = suite.journal.tasks("generation")[0]
    if prepared_kind == "bootstrap":
        task, audit, _, _ = replay_proof(suite, task)
        audit["phase"] = "prepared"
        write_json(suite._directory(task) / "bootstrap-recovery.json", audit)
    else:
        task, intent = unscored_proof(suite, task)
        intent["phase"] = "prepared"
        write_json(suite.directory / "unscored-rehost-intents" / (task["id"] + ".json"), intent)
    assert suite._dispatch_key(task)[0] in (2, 3)
    monkeypatch.setattr(suite, "_execute_task", lambda row: pytest.fail("Prepared recovery guard was bypassed"))
    suite._task(task)


def test_ambiguous_replay_still_fails_actual_worker_before_generation(suite):
    task, audit, _, _ = replay_proof(suite, suite.journal.tasks("generation")[0])
    audit["replay_manifest_sha256"] = "bad"
    write_json(suite._directory(task) / "bootstrap-recovery.json", audit)
    assert suite._dispatch_key(task)[0] == 2
    calls = []
    class FakeBackend:
        job_id = "772379"
        hardware = {}
        def start(self):
            calls.append("attach-existing-fake-session")
        def wait_until_ready(self):
            pass
        def runner(self):
            return object()
        def close(self):
            calls.append("close-fake-session")
        def wait_until_finished(self, timeout):
            return {"terminal": True, "state": "COMPLETED", "exit_code": "0:0", "job_id": self.job_id}
    suite.backend_factory = lambda *args: FakeBackend()
    suite.generate = lambda *args, **kwargs: pytest.fail("Ambiguous replay reached generation")
    suite._task(task)
    fresh = suite.journal.task(task["id"])
    assert fresh["state"] == "recovery_required"
    assert "authenticated allocation and manifest" in fresh["error"]
    assert calls == ["attach-existing-fake-session", "close-fake-session"]
