"""CPU regressions for the explicit recovery control; no network, model or GPU."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest


ROOT = Path(os.environ.get("EVALUATION_RECOVERY_CPU_SOURCE", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/gpu_evaluation_recovery_acceptance.py"
spec = importlib.util.spec_from_file_location("gpu_recovery_acceptance", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def frozen(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    for name in helper.REQUIRED_SOURCE:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    git(source, "add", ".")
    git(source, "-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "CPU source fixture, not an actual publication")
    head = git(source, "rev-parse", "HEAD")
    git(source, "update-ref", "refs/remotes/origin/main", head)
    git(source, "checkout", "--detach", "-q", head)
    return source, head, tmp_path / "control"


def plan(frozen):
    source, head, output = frozen
    return helper.make_plan(source, head, source / "configs/romeo.json", "node-a", "node-a,node-b", output, "cpu-control")


def strict_transport(stdout):
    from optimizer_agent.execution_backend import SSHTransport
    transport = create_autospec(SSHTransport, instance=True, spec_set=True)
    def run(argv, *, data=None, timeout=60, check=True):
        assert data is None and timeout == 60 and check is True
        assert argv[0] in {"sinfo", "squeue"}
        body = stdout(argv) if callable(stdout) else stdout
        return subprocess.CompletedProcess(argv, 0, body.encode("utf-8") if isinstance(body, str) else body, b"")
    transport.run.side_effect = run
    return transport


def test_plan_is_public_no_import_network_or_output_creation(frozen, monkeypatch, capsys):
    source, head, output = frozen
    monkeypatch.setattr(helper, "load_api", lambda *a: pytest.fail("No runtime imports in plan mode"))
    monkeypatch.setattr(helper, "run", lambda *a: pytest.fail("Plan must not execute"))
    assert helper.main(["--source", str(source), "--source-head", head, "--node", "node-a",
                        "--eligible-nodes", "node-b,node-a", "--output", str(output)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["effective_execution_config"]["exclude"] == "node-b"
    assert value["effective_execution_config"]["gres"] == "gpu:h100:1"
    assert value["model_calls"] == value["final_dispatches"] == 0
    assert value["maximum_owned_jobs"] == 2 and not output.exists()
    assert value["source"]["recovery_module_sha256"] == helper.RECOVERY_SHA


@pytest.mark.parametrize("bad", ["dirty", "head", "branch", "unpublished", "symlink", "api"])
def test_frozen_source_guard_refuses_before_runtime_import(frozen, bad):
    source, head, output = frozen
    if bad == "dirty":
        (source / "extra.txt").write_text("fixture")
    elif bad == "head":
        head = "0" * 40
    elif bad == "branch":
        git(source, "checkout", "-qb", "unsafe-fixture")
    elif bad == "unpublished":
        git(source, "update-ref", "-d", "refs/remotes/origin/main")
    elif bad == "symlink":
        alias = source.parent / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    elif bad == "api":
        p = source / "optimizer_agent/evaluation_recovery.py"
        p.write_text("# unreviewed CPU fixture\n")
        git(source, "add", ".")
        git(source, "-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "other fixture")
        head = git(source, "rev-parse", "HEAD")
        git(source, "update-ref", "refs/remotes/origin/main", head)
    with pytest.raises(helper.Refused):
        helper.make_plan(source, head, source / "configs/romeo.json", "node-a", "node-a,node-b", output, "cpu-control")
    assert not output.exists()


@pytest.mark.parametrize("node,nodes", [("node-c", "node-a,node-b"), ("node-a", "node-a,node-a"),
                                        ("node-a", "node-[a-b]"), ("node-a", "node-a;bad")])
def test_placement_must_be_predeclared_individual_nodes(frozen, node, nodes):
    source, head, output = frozen
    with pytest.raises(helper.Refused):
        helper.make_plan(source, head, source / "configs/romeo.json", node, nodes, output, "cpu-control")


@pytest.mark.parametrize("bad", [None, "pin", "head", "filemap", "clean"])
def test_pinned_publication_authority_supports_frozen_clone_refs_without_weakening_pins(frozen, bad):
    source, head, output = frozen
    identity = helper.source_identity(source, head)
    git(source, "update-ref", "-d", "refs/remotes/origin/main")
    receipt = {"kind": "published_source_local_integration_and_exact_sha_ci_release", "head": head,
        "primary_clean": True, "git_blob_exact_files": len(identity["tracked_files_sha256"]),
        "source_pins": identity["tracked_files_sha256"]}
    if bad == "head":
        receipt["head"] = "0" * 40
    elif bad == "filemap":
        receipt["source_pins"] = {**receipt["source_pins"], "configs/romeo.json": "0" * 64}
    elif bad == "clean":
        receipt["primary_clean"] = False
    path = output.parent / "cpu-fixture-release.json"
    helper.write_exclusive(path, receipt)
    pin = "0" * 64 if bad == "pin" else helper.sha(path.read_bytes())
    if bad:
        with pytest.raises(helper.Refused):
            helper.source_identity(source, head, publication_receipt=path, publication_sha=pin)
    else:
        value = helper.make_plan(source, head, source / "configs/romeo.json", "node-a", "node-a,node-b", output,
            "cpu-control", publication_receipt=path, publication_sha=pin)
        assert value["source"]["publication_authority"]["sha256"] == pin
        assert not output.exists()


def test_occupied_intent_cannot_reenter_or_import_runtime(frozen, monkeypatch):
    value = plan(frozen)
    output = Path(value["output"])
    helper.write_exclusive(output / "allocation-intent.json", {"old": "immutable"})
    before = (output / "allocation-intent.json").read_bytes()
    monkeypatch.setattr(helper, "load_api", lambda *a: pytest.fail("No runtime for occupied intent"))
    with pytest.raises(helper.Refused, match="Occupied"):
        helper.run(value)
    assert (output / "allocation-intent.json").read_bytes() == before


def test_durable_exclusive_intent_keeps_old_bytes(tmp_path, monkeypatch):
    observed = []
    original = helper.sync_dir
    monkeypatch.setattr(helper, "sync_dir", lambda p: (observed.append(Path(p)), original(p))[-1])
    path = tmp_path / "new" / "nested" / "intent.json"
    helper.write_exclusive(path, {"step": 1})
    assert observed[:4] == [tmp_path, tmp_path / "new", tmp_path / "new", path.parent]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        helper.write_exclusive(path, {"step": 2})
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [None, "extra", "busy", "malformed"])
def test_scheduler_preflight_is_readonly_exact_and_no_adaptive_placement(bad):
    calls = []
    nodes = "node-a|gpu:h100:4|armgpu|idle\nnode-b|gpu:h100:4|armgpu|idle\n"
    if bad == "extra":
        nodes += "node-c|gpu:h100:4|armgpu|idle\n"
    elif bad == "busy":
        nodes = nodes.replace("armgpu|idle", "armgpu|mix", 1)
    elif bad == "malformed":
        nodes += "unparseable\n"
    def response(argv):
        calls.append(argv)
        return nodes if argv[0] == "sinfo" else "999|protected-other|RUNNING|node-b\n"
    transport = strict_transport(response)
    config = SimpleNamespace(partition="long", constraint="armgpu")
    value = {"predeclared_eligible_nodes": ["node-a", "node-b"], "target_node": "node-a"}
    if bad:
        with pytest.raises(helper.Refused):
            helper.scheduler_preflight(transport, config, value)
        assert len(calls) == 1
    else:
        observed = helper.scheduler_preflight(transport, config, value)
        assert len(calls) == 2 and "protected-other" in observed["user_queue_stdout"]


def test_real_sshtransport_signature_and_captured_bytes_without_any_ssh(monkeypatch):
    from optimizer_agent import execution_backend
    nodes = b"node-a|gpu:h100:4|armgpu|idle\nnode-b|gpu:h100:4|armgpu|idle\n"
    queue = "999|protégé|RUNNING|node-b\n".encode("utf-8")
    commands = []
    def completed(argv, *, input, capture_output, timeout, check):
        assert argv[0] == "ssh" and input is None and capture_output is True and timeout == 60 and check is False
        commands.append(argv[-1])
        assert argv[-1].startswith(("sinfo ", "squeue "))
        return subprocess.CompletedProcess(argv, 0, nodes if argv[-1].startswith("sinfo ") else queue, b"")
    monkeypatch.setattr(execution_backend.subprocess, "run", completed)
    transport = execution_backend.SSHTransport("fixture-does-not-connect")
    observed = helper.scheduler_preflight(transport, SimpleNamespace(partition="long", constraint="armgpu"),
        {"predeclared_eligible_nodes": ["node-a", "node-b"], "target_node": "node-a"})
    assert len(commands) == 2
    assert observed["user_queue_stdout"] == queue.decode("utf-8")
    assert observed["user_queue_stdout_sha256"] == helper.sha(queue)
    assert observed["sinfo_stdout_sha256"] == helper.sha(nodes)
    assert "protégé" in json.dumps(observed, ensure_ascii=False)
    # The same strict autospec that backs all scheduler doubles now rejects
    # the historical helper bug before a permissive fake can conceal it.
    fake = strict_transport(nodes)
    with pytest.raises(TypeError):
        fake.run(["sinfo"], readonly=True)


def test_scheduler_stdout_uses_strict_utf8_without_silent_replacement():
    with pytest.raises(UnicodeDecodeError):
        helper.scheduler_read_text(strict_transport(b"\xff"), ["sinfo"])
    with pytest.raises(helper.Refused, match="stdout"):
        helper.scheduler_read_text(strict_transport(None), ["sinfo"])


@pytest.mark.parametrize("terminal,mismatch", [(False, False), (True, False), (False, True)])
def test_owned_close_uses_real_backend_signatures_and_never_submits_or_cancels(tmp_path, terminal, mismatch):
    from optimizer_agent import suite_orchestrator, campaign_config
    from optimizer_agent.execution_backend import SlurmBackend, SlurmConfig
    transport = strict_transport(b"")
    backend = SlurmBackend(SlurmConfig(), ROOT, tmp_path / "task/worker", "cpu-owned-session", transport=transport)
    task = {"id": "cpu-owned-task", "session": backend.session_id, "job_id": "999" if mismatch else "101",
            "payload": {"config": campaign_config.CampaignConfig().to_dict()}}
    helper.write_exclusive(backend.results_dir / "slurm-session.json", {"session_id": backend.session_id,
        "job_id": "101", "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    def attached():
        backend.job_id = "101"
        backend.historical_only = True
        return {"job_id": "101", "terminal": terminal, "state": "COMPLETED" if terminal else "RUNNING", "exit_code": "0:0"}
    backend.attach_existing = create_autospec(backend.attach_existing, side_effect=attached)
    backend.close = create_autospec(backend.close)
    backend.wait_until_finished = create_autospec(backend.wait_until_finished,
        return_value={"job_id": "101", "terminal": True, "state": "COMPLETED", "exit_code": "0:0"})
    suite = SimpleNamespace(_backend=lambda *args: backend, _directory=lambda selected: tmp_path / "task")
    api = SimpleNamespace(suite_orchestrator=suite_orchestrator)
    if mismatch:
        with pytest.raises(ValueError, match="ownership mismatch"):
            helper.close_owned(api, suite, task)
        backend.close.assert_not_called()
        backend.wait_until_finished.assert_not_called()
    else:
        result = helper.close_owned(api, suite, task)
        assert result["job_id"] == "101" and result["terminal"] is True
        backend.wait_until_finished.assert_called_once_with(timeout=120)
        if terminal:
            backend.close.assert_not_called()
        else:
            backend.close.assert_called_once_with()  # Never cancel=True/scancel.
    assert transport.mock_calls == []


@pytest.mark.parametrize("bad", [None, "config", "directory", "attempt"])
def test_child_authenticates_effective_config_and_remote_identity_before_dispatch(frozen, monkeypatch, bad):
    value = plan(frozen)
    output = Path(value["output"])
    helper.write_exclusive(output / "allocation-intent.json", {"kind": "evaluation_recovery_acceptance_intent", "plan": value})
    config = value["effective_execution_config"]
    helper.write_exclusive(output / "effective-slurm-config.json", config)
    task = {"id": "cpu-task", "attempt": 0, "session": "fixture-owned-session"}
    permit = {**task, "remote_dir": config["remote_root"] + "/sessions/" + task["session"],
              "effective_config_sha256": helper.sha((output / "effective-slurm-config.json").read_bytes())}
    if bad == "config":
        config = {**config, "exclude": "adaptive-other-node"}
        (output / "effective-slurm-config.json").write_text(json.dumps(config))
    elif bad == "directory":
        permit["remote_dir"] += "-wrong"
    elif bad == "attempt":
        permit["attempt"] += 1
    helper.write_exclusive(output / "phase-1-launch-intent.json", permit)
    calls = []
    suite = SimpleNamespace(journal=SimpleNamespace(tasks=lambda kind: [task]), directory=output / "isolated-study",
                            _task=lambda selected: calls.append(selected))
    suite.journal.task = lambda key: {"state": "completed"}
    api = SimpleNamespace(dispatch_attestation=SimpleNamespace(capture_dispatch_attestation=lambda *a: calls.append("epoch")))
    monkeypatch.setattr(helper, "source_identity", lambda *a, **k: {})
    monkeypatch.setattr(helper, "load_api", lambda *a: api)
    monkeypatch.setattr(helper, "new_suite", lambda *a: suite)
    if bad:
        with pytest.raises(helper.Refused):
            helper.child(output / "allocation-intent.json", 1)
        assert calls == []
    else:
        assert helper.child(output / "allocation-intent.json", 1) == 0
        assert calls == ["epoch", task]


def test_import_failure_retains_durable_intent_and_prevents_retry(frozen, monkeypatch):
    value = plan(frozen)
    output = Path(value["output"])
    def unavailable(source):
        assert (output / "allocation-intent.json").exists()
        raise RuntimeError("Declared CPU fixture import failure")
    monkeypatch.setattr(helper, "load_api", unavailable)
    result = helper.run(value)
    assert result["ok"] is False and result["status"] == "incomplete" and result["jobs"] == []
    assert result["uncertain_old_work_preserved"] is None
    manifest = helper.read(output / "proof-sha256.json")
    assert "proof-sha256.json" not in manifest["files"]
    assert "evaluation-recovery-acceptance.json" not in manifest["files"]
    with pytest.raises(helper.Refused, match="Occupied"):
        helper.run(value)


def test_unexpected_terminal_child_closes_only_owned_job_and_never_dispatches_second(frozen, monkeypatch):
    value = plan(frozen)
    from optimizer_agent import campaign_config
    api = SimpleNamespace(campaign_config=campaign_config)
    task = {"id": value["run_id"], "session": "owned-only", "attempt": 0, "job_id": None}
    calls = []
    config = SimpleNamespace(partition="long", constraint="armgpu")
    transport = strict_transport(lambda argv: "node-a|gpu:h100:4|armgpu|idle\nnode-b|gpu:h100:4|armgpu|idle\n" if argv[0] == "sinfo" else "999|protected|RUNNING|node-b\n")
    backend = SimpleNamespace(config=config, transport=transport)
    suite = SimpleNamespace(journal=SimpleNamespace(tasks=lambda *a: [task], task=lambda *a: task),
        _directory=lambda t: Path(value["output"]) / "fixture-campaign", _backend=lambda *a: backend)
    monkeypatch.setattr(helper, "load_api", lambda *a: api)
    monkeypatch.setattr(helper, "new_suite", lambda *a: suite)
    def failed_child(plan, phase, suite):
        calls.append(("child", phase))
        task["job_id"] = "101"
        return 2
    def close(api, selected_suite, owned):
        calls.append(("close_exact", owned["job_id"]))
        assert selected_suite is suite and owned["session"] == "owned-only"
        return {"job_id": "101", "terminal": True, "state": "COMPLETED", "exit_code": "0:0"}
    monkeypatch.setattr(helper, "run_child", failed_child)
    monkeypatch.setattr(helper, "close_owned", close)
    result = helper.run(value)
    assert result["ok"] is False and result["status"] == "incomplete"
    assert calls == [("child", 1), ("close_exact", "101")]
    assert result["jobs"][0]["job_id"] == "101"
    assert (Path(value["output"]) / "postflight.json").is_file()


@pytest.mark.parametrize("bad", [None, "node", "session", "job", "gres"])
def test_actual_ready_placement_refuses_drift_before_runtime_baseline(bad):
    backend = SimpleNamespace(session_id="owned-session", job_id="101", config=SimpleNamespace(gres="gpu:h100:1"),
        hardware={"hostname": "node-a.cluster.invalid", "slurm_job_id": "101", "fingerprint_id": "ready-fixture",
                  "device_properties": {"uuid": "CPU-FIXTURE-NOT-GPU"}})
    if bad == "node":
        backend.hardware["hostname"] = "newly-added-node.cluster.invalid"
    elif bad == "session":
        backend.session_id = "unowned-session"
    elif bad == "job":
        backend.hardware["slurm_job_id"] = "999"
    elif bad == "gres":
        backend.config.gres = "gpu:h100:4"
    if bad:
        with pytest.raises(helper.Refused, match="Actual ready worker"):
            helper.actual_placement(SimpleNamespace(backend=backend), "owned-session", {"target_node": "node-a"})
    else:
        value = helper.actual_placement(SimpleNamespace(backend=backend), "owned-session", {"target_node": "node-a"})
        assert value["job_id"] == "101" and value["actual_hostname"] == "node-a.cluster.invalid"


def test_generation_callback_checks_actual_placement_before_runtime_or_measurement(tmp_path, monkeypatch):
    helper.write_exclusive(tmp_path / "study-plan.json", {"fixture": True})
    captured = {}
    def constructor(plan, directory, *, generate):
        captured["generate"] = generate
        return SimpleNamespace()
    api = SimpleNamespace(suite_orchestrator=SimpleNamespace(SuiteOrchestrator=constructor))
    monkeypatch.setattr(helper, "control_generation", lambda *a, **k: pytest.fail("No runtime/measurement on wrong placement"))
    helper.new_suite(api, {"output": str(tmp_path), "target_node": "node-a"}, phase=1)
    runner = SimpleNamespace(backend=SimpleNamespace(hardware={"hostname": "node-b"}))
    with pytest.raises(helper.Refused, match="Actual ready worker"):
        captured["generate"]({}, runner, tmp_path / "generation", "session")
    assert not (tmp_path / "control-proof").exists()


@pytest.fixture
def cpu_runtime(tmp_path, monkeypatch):
    # Reuse explicit fake runner/evaluator fixtures, while runtime, graph,
    # checkpoint, canonical cache and recovery APIs are the actual code.
    sys.path.insert(0, str(ROOT / "tests"))
    from test_campaign_runtime import RunnerFixture, EvaluatorFixture
    from optimizer_agent import campaign, campaign_config, suite_orchestrator, evaluation_recovery
    assert Path(campaign.__file__).resolve().is_relative_to(ROOT)
    from optimizer_agent.execution_backend import SlurmConfig
    monkeypatch.setattr(campaign, "Evaluator", EvaluatorFixture)
    EvaluatorFixture.instances, EvaluatorFixture.confirmations = [], []
    config = campaign_config.CampaignConfig(run_id="scratch-cpu-recovery", backend="local", max_trials=1,
        experimental_mode="vanilla", research_required=False, web_search_required=False, autotuning_enabled=False,
        max_repairs_per_trial=0, search_sizes=(8,), final_sizes=(11,), profile=False,
        benchmark_series=3, confirmation_batches=3, launches_per_series=2, actions=("coalescing",))
    entry = {"mode": "functional_static_no_model", "config": config.to_dict(), "group_id": "unused-private-final", "seed": 1729, "workload": "matmul"}
    actions = []
    class Backend:
        def __init__(self, task, directory):
            self.task, self.results_dir = task, Path(directory) / "worker"
            self.config, self.session_id, self.job_id = SlurmConfig(), task["session"], None
            self.remote_dir = self.config.remote_root + "/sessions/" + self.session_id
        def attach_existing(self):
            self.job_id = "101"
            actions.append("attach")
            return {"terminal": True, "state": "COMPLETED", "job_id": "101", "exit_code": "0:0"}
        def _json(self, *args, **kwargs):
            return None
        def close(self):
            pytest.fail("CPU recovery fixture must never touch a worker")
    suite = suite_orchestrator.SuiteOrchestrator({"entries": [entry], "modes": ["functional_static_no_model"]},
        tmp_path / "study", backend_factory=lambda t, c, d: Backend(t, d))
    task = suite.journal.tasks("generation")[0]
    directory = suite._directory(task)
    backend = Backend(task, directory)
    campaign.write_json(directory / "worker/slurm-session.json", {"session_id": task["session"], "job_id": "101",
        "remote_dir": backend.remote_dir, "config": asdict(backend.config)})
    api = SimpleNamespace(campaign=campaign, campaign_config=campaign_config, suite_orchestrator=suite_orchestrator,
                          evaluation_recovery=evaluation_recovery)
    return SimpleNamespace(api=api, suite=suite, task=task, entry=entry, runner=RunnerFixture(),
                           RunnerFixture=RunnerFixture, proof=tmp_path / "proof", actions=actions)


def first_crash(value):
    def fault(code):
        assert code == helper.CRASH_EXIT
        raise KeyboardInterrupt("explicit CPU fixture process boundary")
    with pytest.raises(KeyboardInterrupt, match="CPU fixture"):
        helper.control_generation(value.api, value.entry, value.runner,
            value.suite._directory(value.task) / "generation", value.task["session"],
            phase=1, proof_directory=value.proof, crash=fault)
    old = value.suite._directory(value.task) / "generation"
    intents = list((old / "trials").glob("*/evaluation-requests/*/intent.json"))
    assert len(intents) == 1 and not (intents[0].parent / "completion.json").exists()
    boundary = helper.read(value.proof / "crash-boundary.json")
    assert boundary["event_counts_before_exit"].get("tree_reward", 0) == 0
    value.suite.journal.update(value.task["id"], state="recovery_required", job_id="101")
    return old, intents[0]


def test_actual_cpu_graph_cold_copy_replay_no_duplicate_evaluation_or_tell(cpu_runtime):
    value = cpu_runtime
    old, intent = first_crash(value)
    inventory = value.api.evaluation_recovery._inventory(old)
    value.suite.recover_evaluation(value.task["id"], intent.parent.name, "Explicit CPU fixture recovery after measured-but-uncommitted boundary")
    current = value.suite.journal.task(value.task["id"])
    new = value.suite._directory(current) / "generation"
    cost = helper.uncertain_cost_state(new, intent.parent.name, helper.read(intent)["attempt_id"])
    assert cost["uncertain_preserved"] and cost["old_reservation_actual"] is None
    result = helper.control_generation(value.api, value.entry, value.runner, new, current["session"],
        phase=2, proof_directory=value.proof, resume=True)
    assert result["status"] == "functional_static_no_model_recovery_control_complete" and result["model_calls"] == 0
    replay = helper.read(value.proof / "replay-observation.json")
    assert replay["tree_rewards_before_replay"] == replay["tree_rewards_after_replay"] == 1
    assert replay["gpu_rpc_calls_before_replay"] == replay["gpu_rpc_calls_after_replay"]
    assert replay["gpu_rpc_calls_before_evaluation_replay"] == replay["gpu_rpc_calls_after_evaluation_replay"]
    assert replay["previous_request_id"] == intent.parent.name and replay["attempt_id"] != replay["previous_attempt_id"]
    assert value.api.evaluation_recovery._inventory(old) == inventory


def test_actual_cpu_context_mismatch_refuses_before_new_evaluator_work(cpu_runtime):
    value = cpu_runtime
    old, intent = first_crash(value)
    value.suite.recover_evaluation(value.task["id"], intent.parent.name, "Explicit CPU fixture preparation, changed UUID must refuse")
    current = value.suite.journal.task(value.task["id"])
    with pytest.raises(ValueError, match="GPU/toolchain changed"):
        helper.control_generation(value.api, value.entry, value.RunnerFixture("different-physical-UUID"),
            value.suite._directory(current) / "generation", current["session"], phase=2,
            proof_directory=value.proof, resume=True)
    assert not (value.proof / "replay-observation.json").exists()


def test_real_subprocess_exit73_preserves_intent_without_completion_or_reward(tmp_path):
    code = '''
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
sys.path[:0]=[sys.argv[1], str(Path(sys.argv[1])/'tests')]
from optimizer_agent import campaign,campaign_config,suite_orchestrator
from test_campaign_runtime import RunnerFixture,EvaluatorFixture
campaign.Evaluator=EvaluatorFixture
spec=importlib.util.spec_from_file_location('scratch',sys.argv[2]);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
c=campaign_config.CampaignConfig(run_id='cpu-process',backend='local',experimental_mode='vanilla',research_required=False,web_search_required=False,autotuning_enabled=False,max_trials=1,max_repairs_per_trial=0,search_sizes=(8,),final_sizes=(11,),profile=False,benchmark_series=3,confirmation_batches=3,launches_per_series=2)
api=SimpleNamespace(campaign=campaign,suite_orchestrator=suite_orchestrator)
h.control_generation(api,{'config':c.to_dict()},RunnerFixture(),Path(sys.argv[3])/'generation','cpu-only-no-job',phase=1,proof_directory=Path(sys.argv[3])/'proof')
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(SCRIPT), str(tmp_path)], capture_output=True)
    assert result.returncode == helper.CRASH_EXIT, result.stderr.decode()
    intents = list((tmp_path / "generation/trials").glob("*/evaluation-requests/*/intent.json"))
    assert len(intents) == 1 and not (intents[0].parent / "completion.json").exists()
    boundary = helper.read(tmp_path / "proof/crash-boundary.json")
    assert boundary["event_counts_before_exit"].get("tree_reward", 0) == 0
    assert boundary["scope"].startswith("Actual measurement observed")
