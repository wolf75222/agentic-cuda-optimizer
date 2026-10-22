"""Explicit uncertain-evaluation recovery, entirely CPU/mock worker fixtures.

These tests use real registry/checkpoint files and immutable request receipts;
their evaluator and scheduler are declared fixtures, never CUDA/SSH/models.
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from optimizer_agent import campaign, evaluation_recovery as recovery
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.execution_backend import SlurmConfig
from optimizer_agent.suite_orchestrator import SuiteOrchestrator
from test_campaign_runtime import EvaluatorFixture, ModelFixture, RunnerFixture, archive_candidate, selected, tuning_request


def no_activity(*args, **kwargs):
    pytest.fail("Recovery preparation must never submit, close, run GPU work or invoke a model")


@pytest.fixture
def uncertain(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "Evaluator", EvaluatorFixture)
    EvaluatorFixture.instances, EvaluatorFixture.confirmations = [], []
    config = CampaignConfig(run_id="uncertain-unit", backend="local", search_sizes=(8,), final_sizes=(11,),
        max_trials=2, profile=False, benchmark_series=3, confirmation_batches=3, launches_per_series=2)
    entry = {"config": config.to_dict(), "mode": "full", "group_id": "fixture-group", "seed": config.seed,
             "workload": "matmul"}
    plan = {"entries": [entry], "modes": ["full"]}
    actions, worker = [], {"state": "COMPLETED", "quarantine": None, "job": "101"}

    class Backend:
        def __init__(self, task, directory):
            self.task, self.results_dir = task, Path(directory) / "worker"
            self.config, self.session_id, self.job_id = SlurmConfig(), task["session"], None
            self.remote_dir = self.config.remote_root + "/sessions/" + task["session"]
        def attach_existing(self):
            actions.append("authenticate_old_worker")
            self.job_id = worker["job"]
            return {"state": worker["state"], "job_id": self.job_id,
                    "terminal": worker["state"] != "RUNNING", "exit_code": "0:0"}
        def _json(self, name, optional=False):
            actions.append(name)
            assert name == "quarantined.json" and optional
            return worker["quarantine"]
        start = runner = close = wait_until_ready = no_activity

    suite = SuiteOrchestrator(plan, tmp_path / "study", backend_factory=lambda t, c, d: Backend(t, d),
                              generate=no_activity, qualify=no_activity)
    task = suite.journal.tasks("generation")[0]
    directory = suite._directory(task)
    runner, model = RunnerFixture(), ModelFixture()
    runtime = campaign.CampaignRuntime(config, runner, directory / "generation", model_backend=model,
                                       generation_only=True)
    state = selected(runtime)
    tuning_request(runtime, state, values=(16,))
    original = runtime.evaluator.benchmark
    monkeypatch.setattr(runtime.evaluator, "benchmark", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("fixture host crash")))
    with pytest.raises(KeyboardInterrupt, match="fixture host crash"):
        runtime.evaluate(state)
    monkeypatch.setattr(runtime.evaluator, "benchmark", original)
    requests = list((runtime.trial_dir(state) / "evaluation-requests").iterdir())
    assert len(requests) == 1 and not (requests[0] / "completion.json").exists()
    request_id = requests[0].name
    # Opaque database contents: preparation must copy bytes, not deserialize RNG.
    with sqlite3.connect(runtime.directory / "langgraph.sqlite") as db:
        db.execute("CREATE TABLE opaque_checkpoint(payload BLOB)")
        db.execute("INSERT INTO opaque_checkpoint VALUES(?)", (b"opaque RNG/checkpoint fixture; never decoded",))
    call_id = "1" * 32
    campaign.write_json(runtime.directory / "model" / (call_id + ".json"),
        {"call_id": call_id, "status": "completed", "returncode": 0, "usage": {"input_tokens": 7, "output_tokens": 2}})
    backend = Backend(task, directory)
    campaign.write_json(directory / "worker/slurm-session.json", {"session_id": task["session"], "job_id": "101",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    suite.journal.update(task["id"], state="recovery_required", job_id="101", error="uncertain fixture evaluation")
    task = suite.journal.task(task["id"])
    return suite, task, runtime, state, config, runner, model, request_id, worker, actions


def prepare(value):
    suite, task, *_, request_id, worker, actions = value
    return suite.recover_evaluation(task["id"], request_id,
        "Explicitly recover the fixture host crash after the immutable evaluation intent")


def resumed_runtime(value):
    suite, task, old, state, config, runner, model, request_id, worker, actions = value
    current = suite.journal.task(task["id"])
    return campaign.CampaignRuntime(config, runner, suite._directory(current) / "generation",
        model_backend=model, generation_only=True)


def test_explicit_recovery_creates_new_attempt_preserving_uncertain_cost_and_single_tell(uncertain):
    suite, task, old, state, config, runner, model, request_id, worker, actions = uncertain
    before = recovery._inventory(old.directory)
    old_intent = campaign.read_json(old.trial_dir(state) / "evaluation-requests" / request_id / "intent.json")
    with pytest.raises(RuntimeError, match="unfinished"):
        old.evaluate(state)
    audit = prepare(uncertain)
    current = suite.journal.task(task["id"])
    assert current["attempt"] == task["attempt"] + 1 and current["session"] != task["session"]
    assert current["state"] == "pending" and current["job_id"] is None
    assert audit["previous_cost_status"] == "uncertain_preserved" and audit["new_job_submitted"] is False
    assert before == recovery._inventory(old.directory)
    recovery.validate_task_recovery(suite, current)
    resumed = resumed_runtime(uncertain)
    assert resumed.evaluator.benchmarks == []  # Complete baseline reused, no new context measurement.
    result = resumed.evaluate(state)
    completed = list((resumed.trial_dir(state) / "evaluation-requests").glob("*/completion.json"))
    assert len(completed) == 1
    request, completion, measurement = campaign.evaluation_record(completed[0].parent)
    previous_request = campaign.read_json(old.trial_dir(state) / "evaluation-requests" / request_id / "request.json")
    assert recovery._relocated_request(request, resumed.directory, old.directory) == previous_request
    new_manifest_paths = [name for manifest in request["compilation_inputs"].values() for name in manifest["files"]]
    assert any(Path(name).is_relative_to(resumed.directory) for name in new_manifest_paths)
    assert not any(Path(name).is_relative_to(old.directory) for name in new_manifest_paths)
    assert request["recovery"]["previous_request_id"] == request_id
    assert completion["attempt_id"] != old_intent["attempt_id"]
    assert request["recovery"]["previous_attempt_id"] == old_intent["attempt_id"]
    with resumed.store.connection() as db:
        attempt = db.execute("SELECT status,payload FROM attempts WHERE attempt_id=?", (old_intent["attempt_id"],)).fetchone()
        reservation = db.execute("SELECT actual FROM reservations WHERE reservation_id=?", ("evaluation-" + request_id,)).fetchone()
    assert attempt["status"] == "running" and reservation["actual"] is None
    assert json.loads(attempt["payload"])["evaluation_request_id"] == request_id
    benchmarks = list(resumed.evaluator.benchmarks)
    assert resumed.evaluate({**state, **result})["evaluation_id"] == result["evaluation_id"]
    assert resumed.evaluator.benchmarks == benchmarks
    resumed.integrate({**state, **result})
    resumed.integrate({**state, **result})
    assert len([e for e in resumed.store.events(config.run_id) if e["kind"] == "tree_reward"]) == 1
    assert before == recovery._inventory(old.directory) and model.calls == []
    assert actions == ["authenticate_old_worker", "quarantined.json"]


@pytest.mark.parametrize("state,quarantine", [
    ("RUNNING", None), ("RUNNING", {}), ("RUNNING", {"worker_poisoned": True}),
    ("ACCOUNTING_PENDING", {"worker_poisoned": True, "slurm_job_id": "wrong"})])
def test_live_or_ambiguous_old_worker_never_authorizes_new_attempt(uncertain, state, quarantine):
    suite, task, old, *_, worker, actions = uncertain
    worker.update(state=state, quarantine=quarantine)
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="neither stopped"):
        prepare(uncertain)
    assert suite.journal.task(task["id"]) == task
    assert not (suite.directory / "evaluation-recovery-intents").exists()


def test_authenticated_remote_quarantine_can_authorize_separate_session(uncertain):
    suite, task, old, *_, worker, actions = uncertain
    worker.update(state="RUNNING", quarantine={"worker_poisoned": True, "slurm_job_id": "101",
        "worker_reservation": {"session_id": task["session"], "job_id": "101"}})
    prepare(uncertain)
    current = suite.journal.task(task["id"])
    marker = campaign.read_json(current["payload"]["source_recovery"]["marker_path"])
    assert marker["worker_proof_basis"] == "authenticated_remote_worker_quarantine"
    assert worker["state"] == "RUNNING"  # Preparation never closes or cancels the old job.


@pytest.mark.parametrize("field,value", [("job_id", "999"), ("session_id", "other"), ("remote_dir", "/other")])
def test_ownership_mismatch_refuses_preparation(uncertain, field, value):
    suite, task, old, *_ = uncertain
    path = suite._directory(task) / "worker/slurm-session.json"
    campaign.write_json(path, {**campaign.read_json(path), field: value})
    with pytest.raises(ValueError, match="ownership"):
        prepare(uncertain)
    assert suite.journal.task(task["id"]) == task


@pytest.mark.parametrize("status,launch", [("prepared", {"state": "prepared"}),
    ("running", {"state": "started", "child_pid": 12345, "started_at_unix": 1}),
    ("failed", {"state": "started", "child_pid": 12345})])
def test_ambiguous_model_child_refuses_without_pid_or_prompt_inspection(uncertain, status, launch):
    suite, task, old, *_ = uncertain
    path = old.directory / "model" / ("2" * 32 + ".json")
    campaign.write_json(path, {"call_id": path.stem, "status": status, "cli_launch": launch})
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="CLI may still be active"):
        prepare(uncertain)
    assert suite.journal.task(task["id"]) == task


def test_cli_output_without_terminal_receipt_refuses(uncertain):
    suite, task, old, *_ = uncertain
    (old.directory / "model" / ("3" * 32 + ".jsonl")).write_text('{"type":"thread.started"}\n')
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="no terminal"):
        prepare(uncertain)


def test_completed_old_request_uses_normal_replay_instead_of_recovery(uncertain):
    suite, task, old, state, config, runner, model, request_id, *_ = uncertain
    campaign.write_json(old.trial_dir(state) / "evaluation-requests" / request_id / "completion.json", {"fixture": "late completion"})
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="normal result replay"):
        prepare(uncertain)
    assert suite.journal.task(task["id"]) == task


def test_old_late_result_is_archived_and_cannot_replace_recovered_measurement(uncertain):
    suite, task, old, state, config, runner, model, request_id, *_ = uncertain
    prepare(uncertain)
    resumed = resumed_runtime(uncertain)
    # Both old physical history and a copied late completion are historical.
    for root in (old.directory, resumed.directory):
        campaign.write_json(root / "trials" / state["trial_id"] / "evaluation-requests" / request_id / "completion.json",
                            {"evaluation_id": "late-not-accepted", "fixture": True})
    result = resumed.evaluate(state)
    assert result["evaluation_id"] != "late-not-accepted"
    assert campaign.read_json(old.trial_dir(state) / "evaluation-requests" / request_id / "completion.json")["fixture"]


@pytest.mark.parametrize("change", ["proposal", "intent", "commit", "marker", "environment"])
def test_runtime_refuses_recovery_pin_or_environment_mismatch_before_evaluation(uncertain, change):
    suite, task, old, state, config, runner, model, request_id, *_ = uncertain
    prepare(uncertain)
    current = suite.journal.task(task["id"])
    destination = suite._directory(current) / "generation"
    marker_path = Path(current["payload"]["source_recovery"]["marker_path"])
    marker = campaign.read_json(marker_path)
    if change == "environment":
        with pytest.raises(ValueError, match="GPU/toolchain changed"):
            campaign.CampaignRuntime(config, RunnerFixture("different-GPU"), destination, model_backend=model,
                                     generation_only=True)
        return
    if change == "proposal":
        path = destination / "trials" / state["trial_id"] / "proposal-0-1.json"
    elif change == "intent":
        path = destination / "trials" / state["trial_id"] / "evaluation-requests" / request_id / "intent.json"
    elif change == "commit":
        path = destination / "evaluation-recoveries" / ("committed-" + marker["epoch_id"] + ".json")
    else:
        path = marker_path
    campaign.write_json(path, {**campaign.read_json(path), "tampered_fixture": True})
    resumed = resumed_runtime(uncertain)
    with pytest.raises((ValueError, RuntimeError)):
        resumed.evaluate(state)
    assert resumed.evaluator.benchmarks == []


def test_interrupted_copy_is_preserved_and_resumed_without_new_epoch_or_worker(uncertain, monkeypatch):
    suite, task, old, state, config, runner, model, request_id, worker, actions = uncertain
    copytree = recovery.shutil.copytree
    def interrupted(source, destination, **kwargs):
        Path(destination).mkdir(parents=True)
        (Path(destination) / "partial.txt").write_text("partial cold copy")
        raise OSError("fixture copy interruption")
    monkeypatch.setattr(recovery.shutil, "copytree", interrupted)
    with pytest.raises(OSError, match="copy interruption"):
        prepare(uncertain)
    path = recovery._intent_path(suite, task["id"], 1)
    audit, raw = campaign.read_json(path), path.read_bytes()
    assert suite.journal.task(task["id"]) == task
    assert recovery.preparation_pending(suite, task)
    monkeypatch.setattr(recovery.shutil, "copytree", copytree)
    receipt = suite.resume_evaluation_preparation(task["id"])
    assert path.read_bytes() == raw and receipt["epoch_id"] == audit["epoch_id"]
    assert suite.journal.task(task["id"])["session"] == audit["new_session"]
    archives = list(Path(audit["new_generation"]).parent.glob("interrupted-evaluation-copy-*"))
    directories = [p for p in archives if p.is_dir()]
    assert len(directories) == 1 and (directories[0] / "partial.txt").read_text() == "partial cold copy"
    assert suite.resume_evaluation_preparation(task["id"]) == receipt


def test_journal_commit_interruption_blocks_dispatch_until_same_preparation_finishes(uncertain, monkeypatch):
    suite, task, old, state, config, runner, model, request_id, worker, actions = uncertain
    write = recovery._write
    def interrupted(path, value):
        if Path(path).name.startswith("committed-"):
            raise OSError("fixture commit receipt interruption")
        return write(path, value)
    monkeypatch.setattr(recovery, "_write", interrupted)
    with pytest.raises(OSError, match="commit receipt"):
        prepare(uncertain)
    current = suite.journal.task(task["id"])
    assert current["state"] == "pending" and current["attempt"] == 1
    assert recovery.preparation_pending(suite, current)
    suite._execute_task = no_activity
    suite._task(current)  # No scheduler/model work, even though journal is pending.
    with pytest.raises(FileNotFoundError):
        recovery.validate_task_recovery(suite, current)
    monkeypatch.setattr(recovery, "_write", write)
    receipt = suite.resume_evaluation_preparation(task["id"])
    assert receipt["new_session"] == current["session"]
    recovery.validate_task_recovery(suite, suite.journal.task(task["id"]))
    with suite.journal.connection() as db:
        assert db.execute("SELECT count(*) FROM task_attempts WHERE task_id=?", (task["id"],)).fetchone()[0] == 1


def test_late_source_result_during_unfinished_preparation_refuses_copy(uncertain, monkeypatch):
    suite, task, old, state, config, runner, model, request_id, *_ = uncertain
    copytree = recovery.shutil.copytree
    monkeypatch.setattr(recovery.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        prepare(uncertain)
    campaign.write_json(old.trial_dir(state) / "evaluation-requests" / request_id / "completion.json", {"late": True})
    monkeypatch.setattr(recovery.shutil, "copytree", copytree)
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="cold campaign changed"):
        suite.resume_evaluation_preparation(task["id"])
    assert suite.journal.task(task["id"]) == task


def test_active_target_process_lock_refuses_even_with_terminal_scheduler_fixture(uncertain):
    suite, task, *_ = uncertain
    lock = suite.directory / "task-locks" / (hashlib.sha256(task["id"].encode()).hexdigest() + ".lock")
    lock.parent.mkdir(exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(recovery.EvaluationRecoveryRefused, match="still active"):
            prepare(uncertain)


def test_cold_copy_preserves_committed_wal_pages_without_decoding_checkpoint_values(uncertain, tmp_path):
    suite, task, old, *_ = uncertain
    source = tmp_path / "wal-fixture.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE public_fixture_counter(value INTEGER)")
        db.execute("INSERT INTO public_fixture_counter VALUES(42)")
        db.commit()
        frozen = {suffix: Path(str(source) + suffix).read_bytes() for suffix in ("", "-wal", "-shm")}
    for suffix, raw in frozen.items():
        (old.directory / ("cold-crash.sqlite" + suffix)).write_bytes(raw)
    before = recovery._inventory(old.directory)
    prepare(uncertain)
    current = suite.journal.task(task["id"])
    destination = suite._directory(current) / "generation"
    for suffix, raw in frozen.items():
        assert (destination / ("cold-crash.sqlite" + suffix)).read_bytes() == raw
    with sqlite3.connect(destination / "cold-crash.sqlite") as db:
        assert db.execute("SELECT value FROM public_fixture_counter").fetchone()[0] == 42
    assert recovery._inventory(old.directory) == before


@pytest.mark.parametrize("alias", ["symlink_file", "symlink_directory", "hardlink"])
def test_aliased_cold_evidence_is_refused_without_copying(uncertain, tmp_path, alias):
    suite, task, old, *_ = uncertain
    target = tmp_path / "public-alias-fixture"
    if alias == "symlink_directory":
        target.mkdir()
        (old.directory / "alias").symlink_to(target, target_is_directory=True)
    else:
        target.write_bytes(b"synthetic public evidence")
        if alias == "hardlink":
            os.link(target, old.directory / "alias")
        else:
            (old.directory / "alias").symlink_to(target)
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="[Aa]liased|[Ss]ymlinked"):
        prepare(uncertain)
    assert suite.journal.task(task["id"]) == task


def test_copy_changed_after_journal_commit_cannot_be_activated(uncertain, monkeypatch):
    suite, task, *_ = uncertain
    write = recovery._write
    def interrupted(path, value):
        if Path(path).name.startswith("committed-"):
            raise OSError("activation interrupted")
        return write(path, value)
    monkeypatch.setattr(recovery, "_write", interrupted)
    with pytest.raises(OSError):
        prepare(uncertain)
    current = suite.journal.task(task["id"])
    (suite._directory(current) / "generation/langgraph.sqlite").write_bytes(b"changed fixture checkpoint")
    monkeypatch.setattr(recovery, "_write", write)
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="changed before activation"):
        suite.resume_evaluation_preparation(task["id"])
    assert recovery.preparation_pending(suite, current)


def test_recovery_task_payload_binding_mismatch_refuses_before_dispatch(uncertain):
    suite, task, *_ = uncertain
    prepare(uncertain)
    current = suite.journal.task(task["id"])
    suite._execute_task = no_activity
    altered = {**current, "payload": {**current["payload"], "mode": "no_pmpp"}}
    with suite.journal.connection() as db:
        db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps(altered["payload"]), task["id"]))
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="configuration mismatch"):
        suite._task(suite.journal.task(task["id"]))


def test_final_list_payload_is_outside_generation_recovery_without_reading_evidence(monkeypatch):
    monkeypatch.setattr(recovery, "_intent_path", no_activity)
    monkeypatch.setattr(recovery, "_read", no_activity)
    task = {"kind": "final", "payload": [{"mode": "full"}, {"mode": "native_codex_vanilla"}]}
    assert recovery.preparation_pending(object(), task) is False
    assert recovery.validate_task_recovery(object(), task) is None


@pytest.mark.parametrize("payload", [[], {"source_recovery": []}])
def test_malformed_generation_payload_is_still_refused(payload):
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="Malformed generation"):
        recovery.validate_task_recovery(object(), {"kind": "generation", "payload": payload})


def test_recovering_historical_ancestor_refuses_before_intent_copy_worker_or_journal_mutation(uncertain, monkeypatch):
    suite, task, old, state, config, runner, model, first_request, worker, actions = uncertain
    prepare(uncertain)
    resumed = resumed_runtime(uncertain)
    monkeypatch.setattr(resumed.evaluator, "benchmark", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("second host crash")))
    with pytest.raises(KeyboardInterrupt):
        resumed.evaluate(state)
    base, _ = resumed._evaluation_request(state, resumed.candidate(state["candidate_id"]))
    tip_request = campaign.digest(recovery.recovery_request(resumed.directory, base))
    current = suite.journal.task(task["id"])
    backend = suite._backend(current, config, suite._directory(current))
    campaign.write_json(backend.results_dir / "slurm-session.json", {"session_id": current["session"], "job_id": "102",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    suite.journal.update(task["id"], state="recovery_required", job_id="102")
    worker["job"] = "102"
    current = suite.journal.task(task["id"])
    before, observed = recovery._inventory(resumed.directory), list(actions)
    preparation = recovery._intent_path(suite, task["id"], current["attempt"] + 1)
    destination = suite._directory({**current, "attempt": current["attempt"] + 1})
    with pytest.raises(recovery.EvaluationRecoveryRefused, match="already authorized"):
        suite.recover_evaluation(task["id"], first_request, "Mistakenly requesting the already recovered historical ancestor")
    assert not preparation.exists() and not destination.exists()
    assert suite.journal.task(task["id"]) == current
    assert recovery._inventory(resumed.directory) == before and actions == observed
    with suite.journal.connection() as db:
        assert db.execute("SELECT count(*) FROM task_attempts WHERE task_id=?", (task["id"],)).fetchone()[0] == 1
    # The rejected typo cannot strand preparation of the actual unresolved tip.
    receipt = suite.recover_evaluation(task["id"], tip_request, "Recover the actual second uncertain evaluation after the host crash")
    assert receipt["previous_request_id"] == tip_request and receipt["new_task_attempt"] == 2
    assert receipt["new_job_submitted"] is False and model.calls == []


def test_second_uncertain_recovery_appends_epoch_and_preserves_both_earlier_attempts(uncertain, monkeypatch):
    suite, task, old, state, config, runner, model, first_request, worker, actions = uncertain
    first = prepare(uncertain)
    resumed = resumed_runtime(uncertain)
    monkeypatch.setattr(resumed.evaluator, "benchmark", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("second host crash")))
    with pytest.raises(KeyboardInterrupt):
        resumed.evaluate(state)
    original_request, _ = resumed._evaluation_request(state, resumed.candidate(state["candidate_id"]))
    second_request = campaign.digest(recovery.recovery_request(resumed.directory, original_request))
    assert second_request != first_request
    current = suite.journal.task(task["id"])
    backend = suite._backend(current, config, suite._directory(current))
    campaign.write_json(backend.results_dir / "slurm-session.json", {"session_id": current["session"], "job_id": "102",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    suite.journal.update(task["id"], state="recovery_required", job_id="102")
    worker["job"] = "102"
    before = recovery._inventory(resumed.directory)
    second = suite.recover_evaluation(task["id"], second_request, "Another fixture crash after the recovered evaluation intent")
    assert second["epoch_id"] != first["epoch_id"] and second["new_task_attempt"] == 2
    assert before == recovery._inventory(resumed.directory)
    latest = resumed_runtime(uncertain)
    result = latest.evaluate(state)
    path = next((latest.trial_dir(state) / "evaluation-requests").glob("*/completion.json"))
    request, complete, _ = campaign.evaluation_record(path.parent)
    assert request["recovery"]["previous_request_id"] == second_request
    with latest.store.connection() as db:
        attempts = db.execute("SELECT attempt_id,status FROM attempts").fetchall()
    assert len(attempts) == 3 and len({row["attempt_id"] for row in attempts}) == 3
    assert sum(row["status"] == "running" for row in attempts) == 2
    assert result["valid"] and model.calls == []


def test_recovery_authorization_does_not_restrict_a_later_scientific_trial(uncertain):
    suite, task, old, state, config, runner, model, *_ = uncertain
    prepare(uncertain)
    resumed = resumed_runtime(uncertain)
    first = resumed.evaluate(state)
    integrated = resumed.integrate({**state, **first})
    next_state = {**state, **integrated, **resumed.select(integrated)}
    assert next_state["trial_id"] != state["trial_id"]
    proposal = campaign.read_json(resumed.trial_dir(state) / "proposal-0-1.json")
    candidate = archive_candidate(resumed, next_state, candidate_id="later-template", source=proposal["source"], parameters={"TILE": 16})
    next_state.update(candidate_id=candidate.candidate_id, repair=0, research_round=1, research_ready=True)
    campaign.write_json(resumed.trial_dir(next_state) / "proposal-0-1.json", proposal)
    result = resumed.evaluate(next_state)
    assert result["valid"]
    request = campaign.read_json(next((resumed.trial_dir(next_state) / "evaluation-requests").glob("*/request.json")))
    assert "recovery" not in request


@pytest.mark.parametrize("command", ["recover-evaluation", "resume-evaluation-preparation"])
def test_explicit_cli_prepares_only_and_never_dispatches(tmp_path, monkeypatch, command, capsys):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/run_quality_suite.py"
    spec = importlib.util.spec_from_file_location("recovery_cli_fixture", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    campaign.write_json(tmp_path / "plan.json", {"entries": []})
    calls = []
    class Suite:
        def __init__(self, *args, **kwargs):
            pass
        def recover_evaluation(self, task, request, reason):
            calls.append((task, request, reason))
            return {"new_job_submitted": False}
        def resume_evaluation_preparation(self, task):
            calls.append((task,))
            return {"new_job_submitted": False}
        def status(self):
            return {"tasks": [], "summary": {"complete": False, "scientific_qualified": False}}
        run = no_activity
    monkeypatch.setattr(script, "SuiteOrchestrator", Suite)
    monkeypatch.setattr(script, "write_report", lambda *a, **k: None)
    args = [command, "--directory", str(tmp_path), "--task", "fixture-task"]
    if command == "recover-evaluation":
        args += ["--request", "a" * 64, "--reason", "Fixture uncertainty after canonical intent"]
    assert script.main(args) == 0 and len(calls) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["jobs_submitted_by_this_command"] is False
