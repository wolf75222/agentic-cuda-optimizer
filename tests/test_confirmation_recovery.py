"""Real runtime/checkpoint/evaluator with an explicitly non-CUDA NumPy runner.

These regressions prove durable orchestration, never CUDA performance. No SSH,
Slurm submission, model process, or historical private campaign is accessed.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from optimizer_agent import campaign, confirmation_recovery as recovery, evaluation_recovery as cold
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.campaign_workflow import build_campaign_graph
from optimizer_agent.execution_backend import SlurmConfig
from optimizer_agent.suite_orchestrator import SuiteOrchestrator
from test_campaign_runtime import ModelFixture, selected
from test_codex_research import proposal_payload
from test_evaluation_recovery import CPUFileRunner


def forbidden(*args, **kwargs):
    pytest.fail("Preparation/replay cannot submit, measure, close or invoke a model")


class NumericalRunner(CPUFileRunner):
    def run(self, run_id, request):
        row = super().run(run_id, request)
        # Declared deterministic timing fixture only; no CUDA timing claimed.
        latency = .5 if "candidate-fixture" in Path(request.source).read_text() else 1.
        row.update(latency_ms=latency, latency_samples_ms=[latency] * request.iterations)
        return row


def checkpoint(runtime, state, node="experiment"):
    with SqliteSaver.from_conn_string(str(runtime.directory / "langgraph.sqlite")) as saver:
        graph = build_campaign_graph(runtime, saver)
        graph.update_state({"configurable": {"thread_id": runtime.config.run_id}}, state, as_node=node)


@pytest.fixture
def interrupted(tmp_path):
    config = CampaignConfig(run_id="confirmation-unit", backend="local", experimental_mode="ablation_no_research",
        search_sizes=(8,), final_sizes=(11,), max_trials=1, profile=False,
        autotuning_enabled=False, benchmark_series=2, confirmation_batches=2, launches_per_series=2)
    worker = {"state": "COMPLETED", "job_id": "101", "terminal": True, "exit_code": "0:0"}
    actions = []

    class Backend:
        def __init__(self, task, directory):
            self.task, self.results_dir = task, Path(directory) / "worker"
            self.config, self.session_id, self.job_id = SlurmConfig(), task["session"], None
            self.remote_dir = self.config.remote_root + "/sessions/" + task["session"]
        def attach_existing(self):
            actions.append("authenticate_old_worker")
            self.job_id = worker["job_id"]
            return dict(worker)
        start = runner = close = wait_until_ready = forbidden

    entry = {"config": config.to_dict(), "mode": "full", "group_id": "fixture", "seed": config.seed, "workload": "matmul"}
    suite = SuiteOrchestrator({"entries": [entry], "modes": ["full"]}, tmp_path / "study",
        backend_factory=lambda t, c, d: Backend(t, d), generate=forbidden, qualify=forbidden)
    task = suite.journal.tasks("generation")[0]
    runner, model = NumericalRunner(tmp_path / "numpy-runner"), ModelFixture()
    runtime = campaign.CampaignRuntime(config, runner, suite._directory(task) / "generation", model_backend=model, generation_only=True)
    state = selected(runtime)
    source = runtime.baseline.source.read_text() + "\n// candidate-fixture\n"
    source_path = runtime.directory / "fixture.cu"
    source_path.write_text(source)
    rule = replace(runtime.baseline.launch_rule, tile_x=16, tile_y=16)
    candidate = campaign.Candidate("fast", source_path, "matmul", rule, {})
    runtime.store.archive_candidate(config.run_id, runtime.candidate_state(candidate,
        tree_parent_id="baseline", source_parent_id="baseline", trial_id=state["trial_id"]))
    state.update(candidate_id=candidate.candidate_id, research_ready=True, needs_research=False, repair=0, research_round=1)
    proposal = proposal_payload()
    proposal.update(parent_id="baseline", source_parent_id="baseline", contract_hash=runtime.contract_hash,
        strategy_family=runtime.trial(state)["action"], source=source, kernel="matmul",
        launch_rule=asdict(candidate.launch_rule),
        parameters=[], references=[], experimental_mode="ablation_no_research")
    campaign.write_json(runtime.trial_dir(state) / "proposal-0-1.json", proposal)
    state.update(runtime.evaluate(state))
    assert state["valid"] is True
    checkpoint(runtime, state)
    confirm = runtime.evaluator.confirm
    runtime.evaluator.confirm = lambda *args: (_ for _ in ()).throw(OSError("fixture initial EDQUOT"))
    with pytest.raises(OSError, match="initial EDQUOT"):
        runtime.integrate(state)
    runtime.evaluator.confirm = confirm
    request_id = runtime._evaluation_path(state).parent.name
    backend = Backend(task, suite._directory(task))
    campaign.write_json(suite._directory(task) / "worker/slurm-session.json", {"session_id": task["session"], "job_id": "101",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    suite.journal.update(task["id"], state="recovery_required", job_id="101", error="fixture EDQUOT during confirmation")
    return suite, suite.journal.task(task["id"]), runtime, state, config, runner, model, request_id, worker, actions


def prepare(value):
    suite, task, *_, request_id, worker, actions = value
    return recovery.prepare_confirmation_recovery(suite, task["id"], request_id,
        "Explicit independent full confirmation after fixture storage failure")


def resumed(value):
    suite, task, _, state, config, runner, model, request_id, worker, actions = value
    current = suite.journal.task(task["id"])
    return campaign.CampaignRuntime(config, runner, suite._directory(current) / "generation", model_backend=model, generation_only=True)


def test_prepare_cold_copy_replays_completed_canonical_without_new_activity(interrupted):
    suite, task, old, state, config, runner, model, request_id, worker, actions = interrupted
    before, calls = cold._inventory(old.directory), list(runner.calls)
    projection = recovery.checkpoint_projection(old.directory, request_id)
    assert projection["incumbent"]["candidate_id"] == "baseline" and projection["incumbent"]["version"] == 0
    with pytest.raises(cold.EvaluationRecoveryRefused, match="completed request"):
        cold.prepare_evaluation_recovery(suite, task["id"], request_id, "Never bypass a completed canonical request")
    prepare(interrupted)
    current = suite.journal.task(task["id"])
    recovery.validate_task_recovery(suite, current)
    assert cold._inventory(old.directory) == before and runner.calls == calls and model.calls == []
    runtime = resumed(interrupted)
    assert runner.calls == calls  # Baseline and canonical benchmark both reused.
    recovery.bind_checkpoint(runtime, recovery._checkpoint(runtime.directory, config.run_id))
    candidate, incumbent = runtime.candidate(state["candidate_id"]), runtime.candidate("baseline")
    path = runtime._evaluation_path(state)
    acquired = campaign.read_json(path)
    result = recovery.confirmation(runtime, state, candidate, incumbent, acquired, path)
    assert result["status"] == "promoted"
    assert len(result["pairs"]) == 2 * len(acquired["cases"])
    assert result["confirmation_recovery"]["historical_pairs_used"] == 0
    after_confirmation = list(runner.calls)
    assert recovery.confirmation(runtime, state, candidate, incumbent, acquired, path) == result
    assert runner.calls == after_confirmation
    # Exercise the real normal integration path: epoch result precedes canonical
    # confirmation publication, store promotion and TreeQuest's single reward.
    runtime.integrate(state)
    runtime.integrate(state)
    assert len([e for e in runtime.store.events(config.run_id) if e["kind"] == "promotion"]) == 1
    assert len([e for e in runtime.store.events(config.run_id) if e["kind"] == "tree_reward"]) == 1
    assert runner.calls == after_confirmation
    assert campaign.read_json(path) == acquired
    assert cold._inventory(old.directory) == before and model.calls == []
    old_actions = list(actions)
    recovery.resume_confirmation_preparation(suite, task["id"])
    assert actions == old_actions and runner.calls == after_confirmation


@pytest.mark.parametrize("mutation", ["running", "ownership", "round", "wrong-next", "reward", "source", "proposal", "environment", "final", "model"])
def test_preparation_refuses_unsafe_or_drifted_state(interrupted, mutation):
    suite, task, runtime, state, config, runner, model, request_id, worker, actions = interrupted
    if mutation == "running":
        worker.update(state="RUNNING", terminal=False)
    elif mutation == "ownership":
        worker["job_id"] = "999"
    elif mutation == "round":
        checkpoint(runtime, {**state, "research_round": 2})
    elif mutation == "wrong-next":
        checkpoint(runtime, state, node="integrate")
    elif mutation == "reward":
        with runtime.store.transaction() as db:
            db.execute("INSERT INTO reward_claims VALUES(?,?,?)", (config.run_id, state["evaluation_id"], state["trial_id"]))
    elif mutation == "source":
        runtime.candidate(state["candidate_id"]).source.write_text("changed")
    elif mutation == "proposal":
        (runtime.trial_dir(state) / "proposal-0-1.json").write_text("{}")
    elif mutation == "environment":
        campaign.write_json(runtime.directory / "environment.json", {"uuid": "wrong"})
    elif mutation == "final":
        campaign.write_json(runtime.directory / "holdout-exposed.json", {"exposed": True})
    else:
        campaign.write_json(runtime.directory / "model" / ("1" * 32 + ".json"), {"call_id": "1" * 32, "status": "running"})
    calls = list(runner.calls)
    with pytest.raises((ValueError, KeyError)):
        prepare(interrupted)
    assert suite.journal.task(task["id"]) == task and runner.calls == calls and model.calls == []


def test_interrupted_copy_and_committed_replay_preserve_prior_attempt(interrupted, monkeypatch):
    suite, task, old, *rest = interrupted
    original = shutil.copytree
    def partial(source, destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "partial").write_text("preserve interrupted copy")
        raise OSError("fixture copy interruption")
    monkeypatch.setattr(shutil, "copytree", partial)
    with pytest.raises(OSError, match="copy interruption"):
        prepare(interrupted)
    assert suite.journal.task(task["id"]) == task and recovery.preparation_pending(suite, task)
    monkeypatch.setattr(shutil, "copytree", original)
    receipt = recovery.resume_confirmation_preparation(suite, task["id"])
    assert receipt["new_job_submitted"] is False
    current = suite.journal.task(task["id"])
    assert list(suite._directory(current).glob("interrupted-evaluation-copy-*/partial"))
    recovery.validate_task_recovery(suite, current)
    assert recovery.resume_confirmation_preparation(suite, task["id"])["replayed"] is True


def test_epoch_failure_never_silently_remeasures(interrupted, monkeypatch):
    prepare(interrupted)
    runtime = resumed(interrupted)
    _, _, _, state, _, runner, _, _, _, _ = interrupted
    path = runtime._evaluation_path(state)
    args = (runtime, state, runtime.candidate(state["candidate_id"]), runtime.candidate("baseline"), campaign.read_json(path), path)
    def failed(*args):
        raise OSError("fixture EDQUOT halfway through confirmation")
    monkeypatch.setattr(runtime.evaluator, "confirm", failed)
    with pytest.raises(OSError, match="EDQUOT"):
        recovery.confirmation(*args)
    monkeypatch.setattr(runtime.evaluator, "confirm", forbidden)
    with pytest.raises(recovery.ConfirmationRecoveryRefused, match="unfinished"):
        recovery.confirmation(*args)
    assert not path.with_name("confirmation.json").exists()


def test_new_worker_uuid_and_incumbent_version_are_strict(interrupted):
    prepare(interrupted)
    _, _, _, state, config, runner, model, _, _, _ = interrupted
    runner.uuid = "other-physical-device"
    with pytest.raises(ValueError, match="GPU/toolchain changed"):
        resumed(interrupted)
    runner.uuid = "fixture-gpu"
    runtime = resumed(interrupted)
    runtime.store.promote(config.run_id, state["candidate_id"], 0, {"status": "improved", "evidence": ["fixture"]})
    path = runtime._evaluation_path(state)
    with pytest.raises(recovery.ConfirmationRecoveryRefused, match="incumbent version"):
        recovery.confirmation(runtime, state, runtime.candidate(state["candidate_id"]), runtime.candidate("baseline"), campaign.read_json(path), path)


def test_nonbaseline_incumbent_version_is_attested_and_used(interrupted):
    _, _, old, state, config, _, _, request_id, _, _ = interrupted
    source = old.directory / "previous.cu"
    source.write_text(old.baseline.source.read_text() + "\n// previous-incumbent-fixture\n")
    previous = campaign.Candidate("previous", source, "matmul", old.baseline.launch_rule, {})
    old.store.archive_candidate(config.run_id, old.candidate_state(previous,
        tree_parent_id="baseline", source_parent_id="baseline", trial_id="earlier-fixture-trial"))
    old.candidate("previous")
    old.store.promote(config.run_id, "previous", 0, {"status": "improved", "evidence": ["prior-fixture-confirmation"]})
    projection = recovery.checkpoint_projection(old.directory, request_id)
    assert projection["incumbent"]["candidate_id"] == "previous" and projection["incumbent"]["version"] == 1
    prepare(interrupted)
    runtime = resumed(interrupted)
    recovery.bind_checkpoint(runtime, recovery._checkpoint(runtime.directory, config.run_id))
    runtime.integrate(state)
    confirmation = campaign.read_json(runtime._evaluation_path(state).with_name("confirmation.json"))
    assert confirmation["incumbent_id"] == "previous" and confirmation["incumbent_version"] == 1
    assert runtime.store.incumbent(config.run_id)["version"] == 2


def test_real_graph_resumes_with_live_sqlite_saver_and_no_new_model(interrupted, monkeypatch):
    prepare(interrupted)
    suite, task, old, state, config, runner, model, _, _, _ = interrupted
    directory = suite._directory(suite.journal.task(task["id"])) / "generation"
    monkeypatch.setattr(campaign, "CodexBackend", lambda *args, **kwargs: model)
    manifest = campaign.run_on_runner(config, runner, directory, resume=True, generation_only=True)
    assert manifest["trials_done"] == 1 and manifest["generation_only"] is True
    assert model.calls == []
    store = campaign.ExperimentStore(directory / "experiments.sqlite")
    assert len([e for e in store.events(config.run_id) if e["kind"] == "tree_reward"]) == 1


def test_epoch_result_fsync_crash_reuses_complete_pairs_without_rpc(interrupted, monkeypatch):
    prepare(interrupted)
    runtime = resumed(interrupted)
    _, _, _, state, _, runner, _, _, _, _ = interrupted
    path = runtime._evaluation_path(state)
    args = (runtime, state, runtime.candidate(state["candidate_id"]), runtime.candidate("baseline"), campaign.read_json(path), path)
    original = campaign._fsync_evaluation_directory
    def fail_result_directory(directory):
        if directory.name.startswith("epoch-") and (directory / "result.json").exists():
            raise OSError("fixture result directory fsync window")
        original(directory)
    monkeypatch.setattr(campaign, "_fsync_evaluation_directory", fail_result_directory)
    with pytest.raises(OSError, match="fsync window"):
        recovery.confirmation(*args)
    calls = list(runner.calls)
    monkeypatch.setattr(campaign, "_fsync_evaluation_directory", original)
    monkeypatch.setattr(runtime.evaluator, "confirm", forbidden)
    assert recovery.confirmation(*args)["status"] == "promoted"
    assert runner.calls == calls


def test_promotion_before_tell_crash_replays_one_canonical_reward(interrupted, monkeypatch):
    prepare(interrupted)
    runtime = resumed(interrupted)
    _, _, _, state, config, runner, _, _, _, _ = interrupted
    tell = runtime.policy.tell_once
    def interrupted_tell(*args, **kwargs):
        raise OSError("fixture crash after promotion before tell")
    monkeypatch.setattr(runtime.policy, "tell_once", interrupted_tell)
    with pytest.raises(OSError, match="before tell"):
        runtime.integrate(state)
    assert runtime.store.incumbent(config.run_id)["version"] == 1
    calls = list(runner.calls)
    recovery.bind_checkpoint(runtime, recovery._checkpoint(runtime.directory, config.run_id))
    monkeypatch.setattr(runtime.policy, "tell_once", tell)
    runtime.integrate(state)
    runtime.integrate(state)
    assert runner.calls == calls
    events = runtime.store.events(config.run_id)
    assert len([e for e in events if e["kind"] == "promotion"]) == 1
    assert len([e for e in events if e["kind"] == "tree_reward"]) == 1


@pytest.mark.parametrize("fault", ["partial", "source-drift", "canonical-drift"])
def test_partial_epoch_or_canonical_drift_never_integrates(interrupted, monkeypatch, fault):
    prepare(interrupted)
    runtime = resumed(interrupted)
    _, _, _, state, _, _, _, _, _, _ = interrupted
    path = runtime._evaluation_path(state)
    candidate, incumbent = runtime.candidate(state["candidate_id"]), runtime.candidate("baseline")
    acquired = campaign.read_json(path)
    if fault == "canonical-drift":
        campaign.write_json(path, {**acquired, "status": "changed"})
    elif fault == "source-drift":
        candidate.source.write_text("changed")
    else:
        confirm = runtime.evaluator.confirm
        def partial(*args):
            result = confirm(*args)
            result["pairs"].pop()
            return result
        monkeypatch.setattr(runtime.evaluator, "confirm", partial)
    with pytest.raises(ValueError):
        recovery.confirmation(runtime, state, candidate, incumbent, acquired, path)
    assert not path.with_name("confirmation.json").exists()


@pytest.mark.parametrize("publication", [".ready.json", ".committed.json"])
def test_preparation_fsync_window_is_resumable(interrupted, monkeypatch, publication):
    suite, task, *_ = interrupted
    original = cold._write
    raised = False
    def fail_after_write(path, value):
        nonlocal raised
        original(path, value)
        if str(path).endswith(publication) and not raised:
            raised = True
            raise OSError("fixture fsync interruption after publication")
    monkeypatch.setattr(cold, "_write", fail_after_write)
    with pytest.raises(OSError, match="fsync interruption"):
        prepare(interrupted)
    monkeypatch.setattr(cold, "_write", original)
    recovery.resume_confirmation_preparation(suite, task["id"])
    recovery.validate_task_recovery(suite, suite.journal.task(task["id"]))
