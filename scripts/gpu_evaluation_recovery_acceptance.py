#!/usr/bin/env python3
"""Functional §11.2 recovery control, excluded from quality campaigns; --plan is default.

No model is used. A declared static proposal traverses the shipped vanilla
generation path, independent oracle and evaluator. Only the control subprocess
exits, immediately after a real measurement and before canonical publication.
Two one-GPU jobs use one predeclared node. Slurm may still choose another GPU:
that is PARTIAL/refused, never a reason to replace UUID or baseline evidence.
The script never retries an occupied intent, recovers an unrelated study,
dispatches FINAL, cancels jobs, or automatically repeats a mismatched lease.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace


RECOVERY_SHA = "106acbb68f7c5be2ee50f230049d01caae4d2c21446c2e141ecfb99413b22fc0"
CRASH_EXIT = 73
REQUIRED_SOURCE = {"optimizer_agent/campaign.py", "optimizer_agent/suite_orchestrator.py",
    "optimizer_agent/evaluation_recovery.py", "optimizer_agent/campaign_workflow.py",
    "optimizer_agent/evaluation.py", "optimizer_agent/execution_backend.py", "optimizer_agent/remote_worker.py",
    "optimizer_agent/benchmark_workloads.py", "examples/naive_matmul.cu", "configs/romeo.json"}
SCOPE = ("functional_static_no_model: isolated functional §11.2 acceptance injection through the internal runtime vanilla path, "
         "not Codex vanilla/native generation; no model, no FINAL and no quality/performance claim. "
         "Old measured-but-uncommitted work remains uncertain; a new allocation must independently match the exact UUID/contract/toolchain.")


class Refused(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise Refused("Expected a JSON object: " + str(path))
    return value


def write_exclusive(path, value):
    """Local intent/receipt only; durable ancestors precede publication."""
    path = Path(path)
    missing, current = [], path.parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        sync_dir(directory.parent)
        sync_dir(directory)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    sync_dir(path.parent)


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE)


def safe_directory(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise Refused("Symlinked paths are forbidden")
    return path.resolve()


def source_identity(source, head, recovery_sha=RECOVERY_SHA, *, publication_receipt=None, publication_sha=None):
    """Public Git files only. No imports, setup, DB or runtime in plan mode."""
    source = safe_directory(source)
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise Refused("A full exact published source HEAD is required")
    if git(source, "rev-parse", "--show-toplevel").decode().strip() != str(source):
        raise Refused("Source must be its own frozen checkout")
    if git(source, "rev-parse", "HEAD").decode().strip() != head:
        raise Refused("Source HEAD mismatch")
    branch = subprocess.run(["git", "-C", str(source), "symbolic-ref", "-q", "HEAD"], capture_output=True)
    if branch.returncode == 0:
        raise Refused("Source must be detached")
    if git(source, "status", "--porcelain", "--untracked-files=all").strip():
        raise Refused("Frozen source is dirty")
    published = subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", head,
                                "refs/remotes/origin/main"], capture_output=True)
    tracked = {}
    for record in git(source, "ls-tree", "-rz", "HEAD").split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        relative = Path(name.decode("utf-8"))
        if relative.is_absolute() or ".." in relative.parts or kind != "blob" or mode not in {"100644", "100755"}:
            raise Refused("Unsafe or nonregular Git source member")
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise Refused("Missing/nonregular frozen source member")
        content = path.read_bytes()
        if sha(content) != sha(git(source, "cat-file", "blob", blob)):
            raise Refused("Source bytes differ from their Git blob")
        if bool(path.stat().st_mode & 0o111) != (mode == "100755"):
            raise Refused("Source mode differs from its Git tree")
        tracked[str(relative)] = sha(content)
    if tracked.get("optimizer_agent/evaluation_recovery.py") != recovery_sha:
        raise Refused("Recovery API is not the reviewed exact module")
    if not REQUIRED_SOURCE <= tracked.keys():
        raise Refused("Frozen source is missing the required runtime/control files")
    authority = None
    if publication_receipt is not None or publication_sha is not None:
        if publication_receipt is None or not isinstance(publication_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", publication_sha):
            raise Refused("An external publication authority requires its exact SHA256")
        path = safe_directory(publication_receipt)
        if not path.is_file() or sha(path.read_bytes()) != publication_sha:
            raise Refused("Publication authority pin mismatch")
        release = read(path)
        if (release.get("kind") != "published_source_local_integration_and_exact_sha_ci_release"
                or release.get("head") != head or release.get("primary_clean") is not True
                or release.get("source_pins") != tracked or release.get("git_blob_exact_files") != len(tracked)):
            raise Refused("Publication authority does not authenticate the exact frozen source filemap")
        authority = {"path": str(path), "sha256": publication_sha, "kind": release["kind"], "head": head}
    elif published.returncode:
        raise Refused("Exact source lacks origin/main ancestry; supply an explicitly pinned publication authority")
    return {"checkout": str(source), "head": head, "dirty": [], "tracked_files_sha256": tracked,
            "publication_basis": "explicitly pinned public release receipt and exact Git filemap" if authority else
                "local refs/remotes/origin/main ancestry; no network refresh",
            "publication_authority": authority, "recovery_module_sha256": recovery_sha}


def node_names(value):
    names = value.split(",") if isinstance(value, str) else value
    if not names or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
        raise Refused("Declare individual node names, without ranges or shell syntax")
    if len(set(names)) != len(names):
        raise Refused("Duplicate node names")
    return sorted(names)


def make_plan(source, head, execution_config, node, eligible_nodes, output, run_id, *, publication_receipt=None, publication_sha=None):
    identity = source_identity(source, head, publication_receipt=publication_receipt, publication_sha=publication_sha)
    source, output = Path(identity["checkout"]), safe_directory(output)
    if output.is_relative_to(source):
        raise Refused("Control evidence must be outside the frozen checkout")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,65}", run_id):
        raise Refused("Unsafe control run ID")
    nodes = node_names(eligible_nodes)
    if node not in nodes:
        raise Refused("The predeclared target must be in the complete eligible-node list")
    config_path = safe_directory(execution_config)
    if not config_path.is_relative_to(source) or not config_path.is_file():
        raise Refused("Execution config must be a tracked frozen source file")
    config_bytes = config_path.read_bytes()
    if identity["tracked_files_sha256"].get(str(config_path.relative_to(source))) != sha(config_bytes):
        raise Refused("Execution config is not source-pinned")
    config = json.loads(config_bytes)
    if not isinstance(config, dict) or not re.fullmatch(r"gpu(?::[A-Za-z0-9_-]+)?:1", str(config.get("gres"))):
        raise Refused("Exactly one GPU per job is required")
    # Explicit individual exclusions only: do not silently reinterpret ranges.
    prior = node_names(config["exclude"]) if config.get("exclude") else []
    if node in prior:
        raise Refused("The target is already excluded by the frozen configuration")
    config["exclude"] = ",".join(sorted((set(nodes) - {node}) | set(prior)))
    return {"schema_version": 1, "kind": "evaluation_recovery_acceptance_plan", "run_id": run_id,
        "scope": SCOPE, "source": identity, "helper_sha256": sha(Path(__file__).read_bytes()),
        "execution_config": {"path": str(config_path), "sha256": sha(config_bytes)},
        "effective_execution_config": config, "target_node": node, "predeclared_eligible_nodes": nodes,
        "output": str(output), "model_calls": 0, "final_dispatches": 0, "maximum_owned_jobs": 2,
        "conditional_success": "Second actual readiness UUID/toolchain must equal first; mismatch means PARTIAL without a third allocation.",
        "stages": ["read-only scheduler preflight and exclusive local intent", "first owned worker + real measurement + controlled process exit73 before completion/tell",
            "authenticate/close only old owned worker and preserve uncertain evidence", "explicit recover-evaluation cold copy/new epoch/request/attempt",
            "second same-config worker, strict UUID check, real evaluation and one tell", "replay completed evaluation/integration without additional RPC/tell", "terminal/protected-job receipts"]}


def load_api(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    campaign = importlib.import_module("optimizer_agent.campaign")
    modules = {name: importlib.import_module("optimizer_agent." + name) for name in
               ("suite_orchestrator", "evaluation_recovery", "dispatch_attestation", "campaign_config", "execution_backend")}
    for module in (campaign, *modules.values()):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise Refused("Already-loaded runtime belongs to another checkout")
    return SimpleNamespace(campaign=campaign, **modules)


def scheduler_read_text(transport, argv):
    """Only the actual SSHTransport.run API; decode its captured byte stdout."""
    result = transport.run(argv, timeout=60)
    if isinstance(result.stdout, bytes):
        return result.stdout.decode("utf-8", errors="strict")
    if isinstance(result.stdout, str):
        return result.stdout
    raise Refused("Scheduler stdout is neither captured bytes nor text")


def scheduler_preflight(transport, config, plan):
    """One bounded inventory, unchanged placement in both jobs; no scheduler writes."""
    result = scheduler_read_text(transport, ["sinfo", "-N", "-h", "-p", config.partition, "-o", "%N|%G|%f|%t"])
    rows = []
    for line in result.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 4 or not re.fullmatch(r"[A-Za-z0-9_.-]+", fields[0]):
            raise Refused("Ambiguous scheduler node inventory")
        rows.append(dict(zip(("node", "gres", "features", "state"), fields)))
    eligible = {row["node"] for row in rows if config.constraint in row["features"].split(",")
                and re.search(r"gpu:h100:[1-9]", row["gres"], re.IGNORECASE)}
    if eligible != set(plan["predeclared_eligible_nodes"]):
        raise Refused("Actual eligible nodes differ from the list declared before the control")
    target = [row for row in rows if row["node"] == plan["target_node"]]
    if len(target) != 1 or target[0]["state"].lower() != "idle":
        raise Refused("Predeclared node must be idle; no other job is moved or stopped")
    queue = scheduler_read_text(transport, ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N"])
    return {"observed_at_unix": time.time(), "nodes": rows,
            "sinfo_stdout_sha256": sha(result.encode("utf-8")), "user_queue_stdout": queue,
            "user_queue_stdout_sha256": sha(queue.encode("utf-8")), "scope": "Point-in-time read-only observation, not continuous protection or scheduling reservation"}


class NoModels:
    def generate_json(self, *args, **kwargs):
        raise Refused("This static acceptance control forbids every model invocation")


class CountingRunner:
    def __init__(self, delegate):
        self.delegate, self.calls = delegate, 0

    def __getattr__(self, key):
        return getattr(self.delegate, key)

    def run(self, run_id, request):
        self.calls += 1
        return self.delegate.run(run_id, request)


def event_counts(runtime):
    # Only this helper's own test database; never select RNG/tree/blob payloads.
    with runtime.store.connection() as db:
        return {row["kind"]: row["count"] for row in db.execute(
            "SELECT kind,COUNT(*) AS count FROM events WHERE run_id=? GROUP BY kind", (runtime.config.run_id,))}


def actual_placement(runner, session, plan):
    """Ready receipt before CampaignRuntime can launch its first baseline."""
    backend = getattr(runner, "backend", None)
    hardware = getattr(backend, "hardware", None)
    hostname = hardware.get("hostname") if isinstance(hardware, dict) else None
    if (not isinstance(hostname, str) or hostname.lower().split(".", 1)[0] != plan["target_node"].lower().split(".", 1)[0]
            or getattr(backend, "session_id", None) != session
            or str(hardware.get("slurm_job_id")) != str(getattr(backend, "job_id", None))
            or not re.fullmatch(r"[1-9][0-9]*", str(getattr(backend, "job_id", None)))
            or not re.fullmatch(r"gpu(?::[A-Za-z0-9_-]+)?:1", str(getattr(getattr(backend, "config", None), "gres", None)))):
        raise Refused("Actual ready worker does not match the predeclared node and exact owned one-GPU session/job")
    return {"target_node": plan["target_node"], "actual_hostname": hostname,
        "hostname_comparison": "case-insensitive first DNS label", "job_id": str(backend.job_id),
        "session_id": session, "gres": backend.config.gres,
        "hardware_fingerprint": hardware.get("fingerprint_id"),
        "device_uuid": hardware.get("device_properties", {}).get("uuid"),
        "scope": "Actual readiness observed before runtime constructor/baseline; no scheduler placement inferred from exclusions alone"}


def control_generation(api, entry, runner, directory, session, *, phase, proof_directory, resume=False,
                       crash=os._exit):
    """Real runtime/graph/evaluator, static proposal; no model or FINAL nodes."""
    c = api.campaign
    directory, proof_directory = Path(directory), Path(proof_directory)
    config = api.suite_orchestrator.config_from_dict(entry["config"])
    counting = CountingRunner(runner)
    runtime = c.CampaignRuntime(config, counting, directory, model_backend=NoModels(), generation_only=True)
    with c.SqliteSaver.from_conn_string(str(directory / "langgraph.sqlite")) as saver:
        graph = c.build_campaign_graph(runtime, saver)
        graph_config = {"configurable": {"thread_id": config.run_id}, "recursion_limit": 100}
        existing = graph.get_state(graph_config)
        if phase == 1:
            if existing.values:
                raise Refused("First phase is never repeated")
            state = {"run_id": config.run_id, "trials_done": 0}
            state.update(runtime.initialize(state))
            state.update(runtime.select(state))
            state.update(runtime.research(state))  # Shipped explicit vanilla policy, not an asserted research guard.
            trial = runtime.trial(state)
            proposal = {"source": runtime.baseline.source.read_text() + "\n// Static §11.2 acceptance control; no optimizer/model claim.\n",
                "kernel": config.workload, "launch_rule": asdict(runtime.baseline.launch_rule), "parameters": {},
                "parent_id": trial["parent"]["candidate_id"], "source_parent_id": trial["parent"]["candidate_id"],
                "contract_hash": runtime.contract_hash, "strategy_family": trial["action"],
                "hypothesis": "Static recovery control, not an optimization hypothesis", "references": [],
                "parameter_space": [], "parameter_bindings": [], "static_acceptance_control": True,
                "model_calls": 0}
            c.write_immutable_json(runtime.trial_dir(state) / "proposal-0-1.json", proposal)
            graph.update_state(graph_config, state, as_node="research")
            measure = runtime.evaluator.benchmark

            def measure_then_exit(candidate, *args, **kwargs):
                result = measure(candidate, *args, **kwargs)
                if result.get("ok") is not True or result.get("status") != "measured":
                    raise Refused("The old real measurement failed; no controlled success is fabricated")
                write_exclusive(proof_directory / "old-measurement-observed.json", result)
                write_exclusive(proof_directory / "crash-boundary.json", {"kind": "controlled_process_exit_after_measurement",
                    "exit_code": CRASH_EXIT, "pid": os.getpid(), "evaluation_id": result["evaluation_id"],
                    "measurement_sha256": sha((proof_directory / "old-measurement-observed.json").read_bytes()),
                    "contract_hash": runtime.contract_hash, "environment": runtime.environment,
                    "event_counts_before_exit": event_counts(runtime), "rpc_calls_in_process": counting.calls,
                    "scope": "Actual measurement observed outside canonical authority; completion/tell not published, cost remains uncertain"})
                crash(CRASH_EXIT)
                raise Refused("Crash hook returned unexpectedly")

            runtime.evaluator.benchmark = measure_then_exit
        elif not existing.values or not existing.next:
            raise Refused("Recovered checkpoint must be unfinished")
        # Stop after the real nested generate/evaluate graph and before diagnose/finalize.
        updates = graph.stream(None, graph_config, stream_mode="updates")
        for update in updates:
            if "experiment" in update:
                break
        else:
            raise Refused("Experiment did not finish normally")
        updates.close()
        state = dict(graph.get_state(graph_config).values)
        if phase == 1:
            raise Refused("The declared old measurement boundary was not reached")
        if state.get("valid") is not True or not state.get("evaluation_id"):
            raise Refused("Recovered evaluation was not independently validated")
        before_evaluation_replay = counting.calls
        replayed = runtime.evaluate(state)
        after_evaluation_replay = counting.calls
        first_integration = runtime.integrate(state)
        before_replay = counting.calls
        rewards = event_counts(runtime).get("tree_reward", 0)
        runtime.integrate(state)
        after_rewards = event_counts(runtime).get("tree_reward", 0)
        if (counting.calls != before_replay or after_evaluation_replay != before_evaluation_replay
                or rewards != 1 or after_rewards != 1
                or replayed.get("evaluation_id") != state["evaluation_id"]):
            raise Refused("Completed replay changed GPU calls, evaluation identity or tell count")
        measurement_path = runtime._evaluation_path(state)
        request, completion, measurement = c.evaluation_record(measurement_path.parent)
        write_exclusive(proof_directory / "replay-observation.json", {"evaluation_id": state["evaluation_id"],
            "request_id": measurement_path.parent.name, "gpu_rpc_calls_before_replay": before_replay,
            "gpu_rpc_calls_after_replay": counting.calls, "tree_rewards_before_replay": rewards,
            "gpu_rpc_calls_before_evaluation_replay": before_evaluation_replay,
            "gpu_rpc_calls_after_evaluation_replay": after_evaluation_replay,
            "tree_rewards_after_replay": after_rewards, "first_integration": first_integration,
            "previous_request_id": request["recovery"]["previous_request_id"],
            "attempt_id": completion["attempt_id"], "previous_attempt_id": request["recovery"]["previous_attempt_id"],
            "contract_hash": runtime.contract_hash, "environment": runtime.environment,
            "measurement_status": measurement["status"], "model_calls": 0, "scope": SCOPE})
        return {"status": "functional_static_no_model_recovery_control_complete", "generation_complete": True,
                "method_status": "completed", "candidate": api.suite_orchestrator.candidate_receipt(runtime.candidate(state["candidate_id"])),
                "model_calls": 0, "final_deferred": True, "excluded_from_quality_cohorts": True, "scope": SCOPE}


def new_suite(api, plan, phase=None):
    def generate(entry, runner, directory, session, **kwargs):
        if phase not in (1, 2):
            raise Refused("Only an explicitly launched control child may generate")
        placement = actual_placement(runner, session, plan)
        write_exclusive(Path(plan["output"]) / "control-proof" / ("phase-%d-placement.json" % phase), placement)
        return control_generation(api, entry, runner, directory, session, phase=phase,
                                  proof_directory=Path(plan["output"]) / "control-proof", **kwargs)
    return api.suite_orchestrator.SuiteOrchestrator(read(Path(plan["output"]) / "study-plan.json"),
        Path(plan["output"]) / "isolated-study", generate=generate)


def child(intent_path, phase):
    intent = read(intent_path)
    if intent.get("kind") != "evaluation_recovery_acceptance_intent":
        raise Refused("Unknown control intent")
    plan = intent["plan"]
    authority = plan["source"].get("publication_authority") or {}
    source_identity(plan["source"]["checkout"], plan["source"]["head"],
                    publication_receipt=authority.get("path"), publication_sha=authority.get("sha256"))
    if plan["helper_sha256"] != sha(Path(__file__).read_bytes()):
        raise Refused("Acceptance helper changed after intent")
    api = load_api(plan["source"]["checkout"])
    suite = new_suite(api, plan, phase)
    task = suite.journal.tasks("generation")[0]
    permit = read(Path(plan["output"]) / ("phase-%d-launch-intent.json" % phase))
    if any(permit.get(key) != task.get(key) for key in ("id", "attempt", "session")):
        raise Refused("Child launch intent differs from current task identity")
    config_path = Path(plan["output"]) / "effective-slurm-config.json"
    if (read(config_path) != plan["effective_execution_config"]
            or sha(config_path.read_bytes()) != permit.get("effective_config_sha256")
            or permit.get("remote_dir") != plan["effective_execution_config"]["remote_root"].rstrip("/") + "/sessions/" + task["session"]):
        raise Refused("Effective placement/configuration changed before child dispatch")
    suite._controller_epoch = api.dispatch_attestation.capture_dispatch_attestation(suite.directory)
    suite._task(task)
    result = suite.journal.task(task["id"])
    return 0 if result["state"] == "completed" else 2


def close_owned(api, suite, task):
    config = api.suite_orchestrator.config_from_dict(task["payload"]["config"])
    backend = suite._backend(task, config, suite._directory(task))
    status = backend.attach_existing()
    if api.suite_orchestrator._local_session_job(task, backend) != str(task["job_id"]):
        raise Refused("Only the exact owned worker may be closed")
    if status.get("terminal") is not True:
        backend.close()  # Exact authenticated session STOP; never scancel.
    terminal = backend.wait_until_finished(timeout=120)
    path = suite._directory(task) / "control-worker-terminal.json"
    if path.exists():
        previous = read(path)
        if str(previous.get("job_id")) != str(task["job_id"]) or previous.get("terminal") is not True:
            raise Refused("Existing control terminal is not the exact owned job")
        return previous
    write_exclusive(path, terminal)
    return terminal


def uncertain_cost_state(generation, request_id, attempt_id):
    """Inspect two public cost/status columns in this NEW control copy only."""
    with sqlite3.connect(Path(generation, "experiments.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        attempt = db.execute("SELECT status FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        reservation = db.execute("SELECT actual FROM reservations WHERE reservation_id=?", ("evaluation-" + request_id,)).fetchone()
    return {"old_attempt_status": attempt[0] if attempt else None,
            "old_reservation_actual": reservation[0] if reservation else "missing",
            "uncertain_preserved": bool(attempt and attempt[0] == "running" and reservation and reservation[0] is None),
            "scope": "No zero actual cost assigned; only the isolated control copy is opened, no RNG/checkpoint values selected"}


def run_child(plan, phase, suite):
    output = Path(plan["output"])
    task = suite.journal.tasks("generation")[0]
    write_exclusive(output / ("phase-%d-launch-intent.json" % phase), {
        **{key: task[key] for key in ("id", "attempt", "session")},
        "remote_dir": plan["effective_execution_config"]["remote_root"].rstrip("/") + "/sessions/" + task["session"],
        "effective_config_sha256": sha((output / "effective-slurm-config.json").read_bytes()),
        "source_head": plan["source"]["head"], "control_method": "functional_static_no_model"})
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child-intent", str(output / "allocation-intent.json"),
               "--child-phase", str(phase)]
    environment = {**os.environ, "PYTHONPATH": plan["source"]["checkout"], "PYTHONDONTWRITEBYTECODE": "1"}
    with (output / ("phase-%d-controller.log" % phase)).open("xb") as log:
        process = subprocess.Popen(command, cwd=plan["source"]["checkout"], env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_exclusive(output / ("phase-%d-controller.json" % phase), {"pid": process.pid, "command": command,
            "source_head": plan["source"]["head"], "model_calls": 0})
        # No global timer or automatic child kill. An interrupt retains intent;
        # operators authenticate child completion before any later cleanup.
        code = process.wait()
    write_exclusive(output / ("phase-%d-controller-terminal.json" % phase), {"pid": process.pid, "returncode": code,
        "ended_at_unix": time.time(), "scope": "Local child completion only; Slurm terminal is separate"})
    return code


def public_requests(directory, expected_job):
    records = []
    for path in sorted((Path(directory) / "request-receipts").glob("*.json")):
        value = read(path)
        report = value.get("report") or {}
        if str(report.get("slurm_job_id")) != str(expected_job):
            raise Refused("Unexpected native request job identity")
        source = Path(value["request"]["source"])
        records.append({"run_id": value["run_id"], "receipt_path": str(path), "receipt_sha256": sha(path.read_bytes()),
            "source_sha256": sha(source.read_bytes()),
            "compiled_source_sha256": sha(report.get("compilation_identity", {}).get("source", "").encode()),
            "uuid": report.get("environment", {}).get("uuid"), "job_id": str(expected_job),
            "guard_readonly_ok": report.get("guard_readonly_ok"), "ok": report.get("ok"),
            "cuda_samples": len(report.get("latency_samples_ms", [])),
            "reported_native_reservation_seconds": report.get("reservation_wall_seconds"), "compile_ms": report.get("compile_ms")})
    return records


def run(plan):
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise Refused("Occupied output is retained; no automatic retry or additional allocation")
    output.mkdir(parents=True, exist_ok=True)
    # The exclusive intent precedes imports with runtime side effects and all SSH/scheduler work.
    write_exclusive(output / "allocation-intent.json", {"kind": "evaluation_recovery_acceptance_intent",
        "created_at_unix": time.time(), "plan": plan, "scope": SCOPE})
    proof = {"kind": "evaluation_recovery_acceptance", "ok": False, "status": "incomplete", "scope": SCOPE,
        "source_head": plan["source"]["head"], "helper_sha256": plan["helper_sha256"], "model_calls": 0,
        "checks": {}, "jobs": [], "uncertain_old_work_preserved": None,
        "control_method": "functional_static_no_model", "excluded_from_quality_cohorts": True}
    api = suite = None
    backend = None
    owned_tasks = []
    try:
        api = load_api(plan["source"]["checkout"])
        config_path = output / "effective-slurm-config.json"
        write_exclusive(config_path, plan["effective_execution_config"])
        c = api.campaign_config.CampaignConfig(run_id=plan["run_id"], execution_config=str(config_path),
            experimental_mode="vanilla", research_required=False, web_search_required=False,
            max_trials=1, max_repairs_per_trial=0, autotuning_enabled=False, actions=("coalescing",),
            search_sizes=(64,), final_sizes=(96,), benchmark_series=3, confirmation_batches=3,
            launches_per_series=2, warmup=1, profile=False, required_sanitizers=())
        entry = {"config": c.to_dict(), "mode": "functional_static_no_model", "group_id": plan["run_id"] + ".unused-final",
                 "seed": c.seed, "workload": c.workload}
        write_exclusive(output / "study-plan.json", {"kind": "isolated_recovery_control", "entries": [entry],
            "modes": ["functional_static_no_model"], "workloads": [c.workload], "global_resource_caps": None,
            "excluded_from_quality_cohorts": True, "scope": SCOPE})
        suite = new_suite(api, plan)
        task = suite.journal.tasks("generation")[0]
        backend = suite._backend(task, c, suite._directory(task))
        write_exclusive(output / "preflight-before-first.json", scheduler_preflight(backend.transport, backend.config, plan))
        code = run_child(plan, 1, suite)
        task = suite.journal.task(task["id"])
        owned_tasks.append(task)
        if code != CRASH_EXIT or not task.get("job_id"):
            raise Refused("First controller did not reach the exact declared process-exit boundary")
        terminal = close_owned(api, suite, task)
        proof["jobs"].append({"phase": 1, "job_id": task["job_id"], "session": task["session"], "terminal": terminal,
                              "allocated_gres": plan["effective_execution_config"]["gres"]})
        old = suite._directory(task) / "generation"
        requests = list((old / "trials").glob("*/evaluation-requests/*/intent.json"))
        if len(requests) != 1 or (requests[0].parent / "completion.json").exists() or (requests[0].parent / "measurement.json").exists():
            raise Refused("Old canonical work is not the single declared unfinished intent")
        old_files = api.evaluation_recovery._inventory(old)
        old_environment = read(old / "environment.json")
        suite.journal.update(task["id"], state="recovery_required", error="Declared static acceptance controller exit after real measurement", finished=time.time())
        write_exclusive(output / "preflight-before-second.json", scheduler_preflight(backend.transport, backend.config, plan))
        recovery = suite.recover_evaluation(task["id"], requests[0].parent.name,
            "Explicit static acceptance recovery after a real measurement and the declared whole-process exit before canonical completion/tell")
        write_exclusive(output / "recovery-api-receipt.json", recovery)
        new_task = suite.journal.task(task["id"])
        cost_state = uncertain_cost_state(suite._directory(new_task) / "generation", requests[0].parent.name,
                                         read(requests[0])["attempt_id"])
        write_exclusive(output / "uncertain-old-cost-observation.json", cost_state)
        proof["uncertain_old_work_preserved"] = api.evaluation_recovery._inventory(old) == old_files and cost_state["uncertain_preserved"]
        proof["checks"]["old_cost_remains_uncertain"] = cost_state["uncertain_preserved"]
        proof["checks"]["preparation_submits_no_job"] = recovery.get("new_job_submitted") is False and new_task.get("job_id") is None
        code = run_child(plan, 2, suite)
        new_task = suite.journal.task(task["id"])
        owned_tasks.append(new_task)
        if new_task.get("job_id"):
            ack = read(suite._directory(new_task) / "worker/slurm-session.json")
            result = new_task.get("result") or read(suite._directory(new_task) / "failure.json")
            terminal = result.get("worker_terminal")
            if not terminal or terminal.get("terminal") is not True:
                terminal = close_owned(api, suite, new_task)
            proof["jobs"].append({"phase": 2, "job_id": new_task["job_id"], "session": new_task["session"],
                                  "terminal": terminal, "allocated_gres": ack["config"]["gres"]})
            hardware_path = suite._directory(new_task) / "worker/hardware.json"
            if hardware_path.exists():
                hardware = read(hardware_path)
                proof["second_worker_hardware"] = {"path": str(hardware_path), "sha256": sha(hardware_path.read_bytes()),
                    "actual_device_identity": api.campaign.device_identity(hardware.get("device_properties", {})),
                    "expected_device_identity": old_environment}
        if code != 0:
            failure_path = suite._directory(new_task) / "failure.json"
            failure = read(failure_path) if failure_path.exists() else {}
            proof["status"] = "partial_same_uuid_reallocation_refused" if "GPU/toolchain changed" in failure.get("error", "") else "incomplete"
            raise Refused(failure.get("error") or "Second controller did not finish the recovery control")
        replay = read(output / "control-proof/replay-observation.json")
        old_intent = read(requests[0])
        new_generation = suite._directory(new_task) / "generation"
        completion_paths = list((new_generation / "trials").glob("*/evaluation-requests/*/completion.json"))
        if len(completion_paths) != 1:
            raise Refused("Expected exactly one new authoritative completed request")
        new_request, completion, measurement = api.campaign.evaluation_record(completion_paths[0].parent)
        old_task_directory = suite._directory(task)
        old_reports = public_requests(old_task_directory, task["job_id"])
        new_reports = public_requests(suite._directory(new_task), new_task["job_id"])
        checks = proof["checks"]
        checks.update(old_process_exit_exact=read(output / "phase-1-controller-terminal.json")["returncode"] == CRASH_EXIT,
            old_measurement_before_tell=read(output / "control-proof/crash-boundary.json")["event_counts_before_exit"].get("tree_reward", 0) == 0,
            actual_predeclared_placement_before_measurement=all(
                read(output / ("control-proof/phase-%d-placement.json" % phase))["target_node"] == plan["target_node"]
                and str(read(output / ("control-proof/phase-%d-placement.json" % phase))["job_id"]) == str(proof["jobs"][phase-1]["job_id"])
                for phase in (1, 2)),
            old_canonical_unfinished_preserved=api.evaluation_recovery._inventory(old) == old_files,
            same_real_uuid_toolchain=read(new_generation / "environment.json") == old_environment == replay["environment"],
            same_contract=new_request["contract_hash"] == read(old / "contract.json")["contract_hash"] == replay["contract_hash"],
            new_epoch_request_attempt=completion_paths[0].parent.name != requests[0].parent.name
                and completion["attempt_id"] != old_intent["attempt_id"]
                and new_request["recovery"]["previous_request_id"] == requests[0].parent.name,
            new_session_job=new_task["session"] != task["session"] and new_task["job_id"] != task["job_id"],
            real_native_both_jobs=bool(old_reports) and bool(new_reports)
                and all(v["ok"] is True and v["guard_readonly_ok"] is True and v["uuid"] == old_environment["uuid"]
                        and v["source_sha256"] == v["compiled_source_sha256"] for v in old_reports + new_reports),
            unique_native_request_ids=len({v["run_id"] for v in old_reports + new_reports}) == len(old_reports + new_reports),
            replay_no_gpu_or_tell=replay["gpu_rpc_calls_before_replay"] == replay["gpu_rpc_calls_after_replay"]
                and replay["gpu_rpc_calls_before_evaluation_replay"] == replay["gpu_rpc_calls_after_evaluation_replay"]
                and replay["tree_rewards_before_replay"] == replay["tree_rewards_after_replay"] == 1,
            independent_oracle_measurement=measurement.get("ok") is True and measurement.get("status") == "measured",
            both_worker_terminal=all(j["terminal"].get("terminal") is True and j["terminal"].get("state") == "COMPLETED"
                and j["terminal"].get("exit_code") == "0:0" and str(j["terminal"].get("job_id")) == str(j["job_id"]) for j in proof["jobs"]))
        write_exclusive(output / "native-request-observations.json", {"old": old_reports, "new": new_reports,
            "scope": "Existing request receipts only; no new sampling. Kernel timing is not a quality comparison; wall times remain separate."})
        write_exclusive(output / "postflight.json", {"observed_at_unix": time.time(), "user_queue_stdout":
            scheduler_read_text(backend.transport, ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N"]),
            "scope": "Point-in-time observation only; no claim of continuous causality/protection"})
        proof.update(ok=all(checks.values()), uncertain_old_work_preserved=checks["old_canonical_unfinished_preserved"] and checks["old_cost_remains_uncertain"],
                     status="evaluation_recovery_acceptance_passed" if all(checks.values()) else "incomplete")
    except Exception as error:
        proof["error"] = type(error).__name__ + ": " + str(error)
        # Only returned/terminal child phases can authorize cleanup. An
        # interrupted parent with a potentially active child never stops it.
        for owned in owned_tasks:
            if not owned.get("job_id") or any(str(row["job_id"]) == str(owned["job_id"]) for row in proof["jobs"]):
                continue
            try:
                terminal = close_owned(api, suite, owned)
                proof["jobs"].append({"job_id": owned["job_id"], "session": owned["session"], "terminal": terminal})
            except Exception as cleanup:
                proof.setdefault("cleanup_errors", []).append(str(cleanup))
    finally:
        if backend is not None and not (output / "postflight.json").exists():
            try:
                queue = scheduler_read_text(backend.transport, ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N"])
                write_exclusive(output / "postflight.json", {"observed_at_unix": time.time(), "user_queue_stdout": queue,
                    "scope": "Point-in-time observation only, including incomplete/refused controls; no continuous causality claim"})
            except Exception as error:
                proof["postflight_error"] = str(error)
        # Manifest excludes itself and the authority receipt, preventing self-reference.
        files = {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob("*"))
                 if p.is_file() and not p.is_symlink() and p.name not in {"proof-sha256.json", "evaluation-recovery-acceptance.json"}
                 and p.suffix != ".bin" and ".sqlite" not in p.name and not p.name.endswith(("-wal", "-shm"))}
        write_exclusive(output / "proof-sha256.json", {"kind": "declared_public_file_integrity", "files": files,
            "scope": "Public/control metadata only, excludes this manifest and authority; opaque DB/numerical inputs not redistributed by this manifest"})
        write_exclusive(output / "evaluation-recovery-acceptance.json", proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-head")
    parser.add_argument("--publication-receipt", type=Path, help="Exact public release receipt when the frozen local clone has different remote-tracking refs")
    parser.add_argument("--publication-receipt-sha256", help="Mandatory SHA256 pin whenever a publication receipt is supplied")
    parser.add_argument("--execution-config", type=Path)
    parser.add_argument("--node")
    parser.add_argument("--eligible-nodes", help="Complete individual eligible nodes declared before any measurement")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="evaluation-recovery-control")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--child-intent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-phase", type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_intent:
        if args.child_phase is None:
            parser.error("Internal child requires its declared phase")
        return child(args.child_intent, args.child_phase)
    if args.run and args.plan:
        parser.error("Choose --plan or --run")
    if any(getattr(args, k) is None for k in ("source", "source_head", "node", "eligible_nodes", "output")):
        parser.error("--source, --source-head, --node, --eligible-nodes and --output are mandatory")
    config = args.execution_config or args.source / "configs/romeo.json"
    plan = make_plan(args.source, args.source_head, config, args.node, args.eligible_nodes, args.output, args.run_id,
                     publication_receipt=args.publication_receipt, publication_sha=args.publication_receipt_sha256)
    result = run(plan) if args.run else plan
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not args.run or result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
