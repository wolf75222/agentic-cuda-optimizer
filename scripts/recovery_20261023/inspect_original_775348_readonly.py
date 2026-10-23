"""Trusted, read-only assessment of the stopped original graph; no recovery.

Default mode describes the protocol without opening a checkpoint.  --inspect
must be reviewed by the primary first.  The only checkpoint API used is
CompiledStateGraph.get_state(); values remain inside this process.  Neither a
graph node, a model, an observer nor a GPU operation is invoked.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import stat
import subprocess

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = "1e9464da54dfc651337a97c3643bfeefec712bc1"
TASK = "quality-20261021-v2.normalization.2718.true_upstream"
SESSION = "suite-gen-81d0d742ded540aeb1343bd846611964"
JOB = "775348"
SOURCE = "85cc10c9d98dcafe08a4d3d66b1f122584f9e8248fe3fa1a1447c70260c0c6cd"
ARTIFACT = "d1b42c3054fb5c86ca9a0edc6d606be442a90006024d45dc0c23f5c5b6256001"
PUBLIC_REFERENCE = "a30cc996eff0f8a6369f4e7ffac1e3e3661fabf7691c6a27f35443596c2e1461"
CODEX_SOURCE_SHA = "326f756ec9c4bb3d9dee0c684c54335ccf6aa28a30dbd8918f310317de52071e"
BRIDGE_SOURCE_SHA = "826906e63e5c69a0b8285d441a7dfcb858d1610c097004f4832917a19fd61241"
NODES = frozenset({"call_model", "tools", "prepare_signature", "research_nvidia",
    "prepare_inputs", "propose", "repair_candidate", "save_kernel", "evaluate_candidate",
    "profile_candidate", "generate_reference", "generate_initial_kernel", "load_reference_kernel",
    "load_initial_kernel", "execute_reference", "prepare_launches", "read_kernel_proposal", "finalize_result"})


class Refused(RuntimeError):
    """Static public code only: never attach private values or nested exceptions."""


def require(condition, code):
    if not condition:
        raise Refused(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    require(".." not in path.parts and path.is_relative_to(root), "outside_allowed_root")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "aliased_evidence")
    return path


def raw(path, root, maximum=32*1024*1024):
    path = safe(path, root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_size <= maximum, "unbounded_or_aliased_evidence")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            body = stream.read(maximum+1)
        after = os.fstat(fd)
        require(len(body) <= maximum and signature(before) == signature(after), "evidence_changed")
        return body
    finally:
        os.close(fd)


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read(path, root):
    return json.loads(raw(path, root))


def proof(path, root):
    body = raw(path, root)
    return {"path": str(Path(path).relative_to(root)), "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body)}


@contextlib.contextmanager
def existing_lock(path, root):
    """Advisory ownership only; open an existing inode read-only, never create."""
    fd = os.open(safe(path, root), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "lock_not_regular")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused("controller_or_task_active") from None
        yield
    finally:
        os.close(fd)


def public_task(study):
    """The suite journal is not a gateway to its private payload."""
    allowed = {"id", "kind", "state", "attempt", "session", "job_id"}
    def authorize(action, table, column, database, origin):
        return sqlite3.SQLITE_OK if (action == sqlite3.SQLITE_SELECT or
            action == sqlite3.SQLITE_READ and table == "tasks" and column in allowed) else sqlite3.SQLITE_DENY
    with contextlib.closing(sqlite3.connect((study/"suite.sqlite").as_uri()+"?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        db.set_authorizer(authorize)
        rows = db.execute("SELECT id,kind,state,attempt,session,job_id FROM tasks WHERE id=?", (TASK,)).fetchall()
    require(len(rows) == 1, "task_missing_or_ambiguous")
    row = dict(zip(("id", "kind", "state", "attempt", "session", "job_id"), rows[0]))
    require(row == {"id": TASK, "kind": "generation", "state": "recovery_required",
                    "attempt": 0, "session": SESSION, "job_id": JOB}, "task_binding_changed")
    return row


def authenticate_original(repo, generation, manifest):
    """Verify all 16 historical .py sources against exact Git blobs; no writes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout
    require(git("rev-parse", UPSTREAM+"^{commit}").decode().strip() == UPSTREAM, "upstream_commit_changed")
    paths = [p for p in git("ls-tree", "-r", "--name-only", UPSTREAM, "optimizer_agent").decode().splitlines()
             if p.endswith(".py")]
    require(len(paths) == 16, "upstream_source_set_changed")
    upstream = manifest["upstream"]
    require(upstream["commit"] == UPSTREAM, "manifest_upstream_changed")
    snapshot = Path(upstream["path"])
    expected = repo.parents[1]/"work/study-source-fefd63c/work"/("original-"+UPSTREAM[:12])
    require(snapshot == expected, "historical_snapshot_path_changed")
    hashes = {}
    for relative in paths:
        expected_raw = git("show", UPSTREAM+":"+relative)
        actual = raw(snapshot/relative, snapshot)
        require(actual == expected_raw, "original_source_bytes_changed")
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    require(upstream["source_hashes"] == hashes and upstream["source_identity"] == digest(hashes),
            "original_source_manifest_changed")
    return snapshot, {"commit": UPSTREAM, "file_count": len(hashes),
        "source_identity": digest(hashes), "source_hashes": hashes}


def compare_cli_result(acknowledged, terminal):
    """Same reply, two documented clocks; retain both instead of rewriting one.

    Pinned CodexBackend records receipt.elapsed_seconds after communicate(), then
    samples CodexResult.elapsed_seconds after event parsing/schema validation.
    All six model/reply fields must match exactly. Durations are independently
    pinned by the raw CLI receipt and the exact ACK/reply record hashes.
    """
    fields = {"payload", "usage", "events", "model", "reasoning_effort", "call_id"}
    require(set(acknowledged) == set(terminal) == fields | {"elapsed_seconds"},
            "original_cli_result_fields_changed")
    require(all(digest(acknowledged[k]) == digest(terminal[k]) for k in fields),
            "original_cli_reply_identity_changed")
    a, t = acknowledged["elapsed_seconds"], terminal["elapsed_seconds"]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (a,t))
            and a >= t, "original_cli_duration_semantics_changed")
    return {"cli_elapsed_sha256": digest(t), "ack_elapsed_sha256": digest(a),
            "ack_sample_not_earlier_than_cli": True,
            "semantics": "CLI receipt: after communicate; ACK result: after parsing/validation; separate historical samples retained"}


def closed_cli_receipts(generation, bridge, manifest):
    """Use the production pure terminal verifier; never construct the transport."""
    folder = generation/"model-transport"
    for filename, expected in (("codex_backend.py", CODEX_SOURCE_SHA), ("upstream_original.py", BRIDGE_SOURCE_SHA)):
        actual = hashlib.sha256(raw(Path(bridge.__file__).parent/filename, Path(bridge.__file__).parent)).hexdigest()
        require(actual == expected == manifest["adapter_source_hashes"][filename],
                "historical_cli_clock_source_changed")
    calls = read(folder/"calls.json", generation)
    require(isinstance(calls, list) and len(calls) == 10, "original_call_count_changed")
    intents = sorted((folder/"intents").glob("*/*/intent.json"))
    require(len(intents) == len(calls), "original_intent_set_changed")
    verifier = object.__new__(bridge.OriginalCodexTransport)
    records, seen = [], set()
    for path in intents:
        intent = read(path, generation)
        request = intent["request"]
        key = bridge.content_hash({"prompt": request["prompt"], "schema": request["schema"],
            "purpose": request["purpose"], "search": request["max_web_search_calls"]})
        require(intent["key"] == key and path.parent.parent.name == key
                and path.parent.name == str(intent["attempt"]), "original_intent_binding_changed")
        require(Path(intent["trace_directory"]) == path.parent/"cli", "original_cli_path_changed")
        trace = safe(path.parent/"cli", generation)
        cli_paths = list(trace.glob("*.json"))
        require(len(cli_paths) == 1, "original_cli_receipt_ambiguous")
        receipt_path = safe(cli_paths[0], generation)
        before = proof(receipt_path, generation)
        result = bridge.OriginalCodexTransport._completed_cli(verifier, path, intent)
        require(before == proof(receipt_path, generation), "original_cli_changed_during_read")
        matching = [row for row in calls if row["key"] == key and row["attempt"] == intent["attempt"]]
        require(len(matching) == 1, "original_call_ack_ambiguous")
        row = matching[0]
        require(row["intent_hash"] == bridge.content_hash(intent)
                and row["record_hash"] == bridge.content_hash({k:v for k,v in row.items() if k != "record_hash"}),
                "original_cli_ack_changed")
        reply = read(path.parent/"reply.json", generation)
        require(reply["intent_hash"] == bridge.content_hash(intent) and reply["record"] == row,
                "original_durable_reply_changed")
        duration = compare_cli_result(row["result"], result)
        require(row["cli_receipt"] == {"path": str(receipt_path), "sha256": before["sha256"]}, "original_cli_pin_changed")
        call_id = result["call_id"]
        require(call_id not in seen, "original_call_id_reused")
        seen.add(call_id)
        records.append({"call_id": call_id, "status": "completed", "returncode": 0, **before,
            "ack_record_hash": row["record_hash"], "durable_reply": proof(path.parent/"reply.json", generation),
            "duration_samples": duration})
    return records


def project_snapshot(snapshot, history, expected_best_source, choose_next):
    """Allowlisted projection: never return a state/config/message/case value."""
    values = snapshot.values
    require(isinstance(values, dict) and values, "checkpoint_state_missing")
    current = values["optimization_history"]
    require(type(values["iteration"]) is int and values["iteration"] == 3
            and type(values["config"].max_iterations) is int and values["config"].max_iterations == 3,
            "checkpoint_iteration_changed")
    require(len(current) == len(history) == 4 and [x.iteration for x in current] == [0,1,2,3],
            "checkpoint_history_changed")
    for attempt, public in zip(current, history):
        require(attempt.iteration == public["iteration"]
                and hashlib.sha256(attempt.kernel_source.encode()).hexdigest()
                    == hashlib.sha256(public["kernel_source"].encode()).hexdigest()
                and len(attempt.case_runs) == len(public["case_runs"]) == 7,
                "checkpoint_attempt_changed")
        require([c.validation_ok for c in attempt.case_runs] == [c["validation_ok"] for c in public["case_runs"]],
                "checkpoint_case_validity_changed")
    best = values["best_attempt"]
    require(best is not None and best.iteration == 2
            and hashlib.sha256(best.kernel_source.encode()).hexdigest() == expected_best_source,
            "checkpoint_selected_artifact_changed")
    require(values.get("error") is None, "checkpoint_has_method_error")
    route = choose_next(values)
    require(route == "finish", "original_route_would_continue")
    upcoming = tuple(snapshot.next)
    require(all(type(n) is str and n in NODES for n in upcoming), "unknown_next_node")
    tasks = tuple(snapshot.tasks)
    require(all(t.name in NODES for t in tasks), "unknown_task_node")
    require(not any(t.error is not None or t.interrupts for t in tasks), "checkpoint_task_error_or_interrupt")
    require(upcoming in {(), ("finalize_result",)} and all(t.name == "finalize_result" for t in tasks),
            "checkpoint_needs_model_or_gpu_node")
    require(tuple(t.name for t in tasks) == upcoming, "next_task_binding_missing")
    return {"next_nodes": list(upcoming), "pending_task_nodes": [t.name for t in tasks],
        "phase": "evaluation_complete_before_finalize" if upcoming else "graph_end",
        "iteration": 3, "max_iterations": 3, "history_count": 4,
        "selected_iteration": 2, "selected_source_sha256": expected_best_source,
        "checkpoint_method_error_present": False, "task_errors_or_interrupts": False,
        "original_choose_next": route, "structural_completion_requires_model_or_gpu": False,
        "permission_to_resume_or_relabel": False}


def readonly_saver(connection):
    """Read the installed LangGraph format through its saver without setup DDL."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    class ReadOnlySaver(SqliteSaver):
        def setup(self):
            if self.is_setup:
                return
            names = {row[0] for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            require({"checkpoints", "writes"} <= names, "checkpoint_schema_missing")
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(writes)")}
            require({"thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx", "channel", "type", "value"} <= columns,
                    "checkpoint_writes_schema_changed")
            self._has_task_path = "task_path" in columns
            self.is_setup = True
        def put(self, *args, **kwargs):
            raise Refused("checkpoint_mutation_forbidden")
        def put_writes(self, *args, **kwargs):
            raise Refused("checkpoint_mutation_forbidden")
    return ReadOnlySaver(connection)


def checkpoint_projection(generation, snapshot_path, config, history, bridge):
    from langgraph.graph import StateGraph
    path = safe(generation/"original-langgraph.sqlite", generation)
    require(not any(Path(str(path)+suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
            "checkpoint_not_cold")
    before = proof(path, generation)
    denied = []
    def authorize(action, table, column, database, origin):
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA and table == "table_info" and column == "writes":
            return sqlite3.SQLITE_OK
        denied.append(action)
        return sqlite3.SQLITE_DENY
    class NoExecution:
        def bind_tools(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            raise Refused("model_or_gpu_execution_forbidden")
    with bridge.original_modules(snapshot_path, generation) as original:
        original_config = original.optimizer_config.OptimizerConfig(generation, 0, config["max_trials"],
            config["warmup"], config["launches_per_series"], "gpt-6-luna", config["max_repairs_per_trial"],
            nvidia_research=False, use_nsight=False, reasoning_effort="max")
        with contextlib.closing(sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorize)
            saver = readonly_saver(db)
            class InspectableGraph(StateGraph):
                def compile(self, *args, **kwargs):
                    kwargs["checkpointer"] = saver
                    return super().compile(*args, **kwargs)
            original.workflow.StateGraph = InspectableGraph
            original.workflow.ChatOpenAI = lambda **kwargs: NoExecution()
            graph = original.workflow.build_graph(NoExecution(), NoExecution(), original_config)
            # get_state does not execute any node. Do not call stream/invoke.
            state = graph.get_state({"configurable": {"thread_id": config["run_id"]}})
            projection = project_snapshot(state, history, SOURCE, original.nodes.choose_next)
    require(not denied and before == proof(path, generation), "checkpoint_write_attempt_or_drift")
    require(not any(Path(str(path)+suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
            "checkpoint_sidecar_created")
    return {**projection, "checkpoint_opaque_sha256": before["sha256"],
        "checkpoint_size": before["bytes"], "checkpoint_api": "CompiledStateGraph.get_state",
        "sqlite_mode": "ro,immutable,query_only; authorizer denies writes; saver setup only validates schema"}


def assess(repo=REPO):
    repo = Path(repo).resolve()
    study = repo/"results/quality-study-v2"
    generation = study/"campaigns"/TASK/"generation"
    with existing_lock(study/"controller.lock", repo), existing_lock(
            study/"task-locks"/(hashlib.sha256(TASK.encode()).hexdigest()+".lock"), repo):
        task = public_task(study)
        reference_path = repo/"work/live/post-model-confirmation-observer-recovery-analysis-fefd63c.json"
        reference_raw = raw(reference_path, repo)
        require(hashlib.sha256(reference_raw).hexdigest() == PUBLIC_REFERENCE, "public_reference_changed")
        reference = json.loads(reference_raw)
        checked = []
        for row in reference["evidence"]:
            relative = row["path"]
            if relative.startswith(str(generation.relative_to(repo))+"/"):
                item = proof(repo/relative, repo)
                require(item["sha256"] == row["sha256"] and item["bytes"] == row["bytes"], "public_evidence_changed")
                checked.append(item)
        required = {"history.json", "config.json", "summary.json", "error.log", "gain-observer/summary.json"}
        require(required <= {str(Path(item["path"]).relative_to(generation.relative_to(repo))) for item in checked},
                "public_reference_incomplete")
        for relative in ("generation/manifest.json", "generation/candidate.json", "generation/generation-progress.json",
                         "generation/frozen-original.cu", "result.json", "worker/slurm-session.json"):
            item = proof(generation.parent/relative, repo)
            if item["path"] not in {row["path"] for row in checked}:
                checked.append(item)
        manifest = read(generation/"manifest.json", repo)
        history = read(generation/"history.json", repo)
        config = read(generation/"config.json", repo)
        progress = read(generation/"generation-progress.json", repo)
        candidate = read(generation/"candidate.json", repo)
        observer = read(generation/"gain-observer/summary.json", repo)
        result = read(generation.parent/"result.json", repo)
        ack = read(generation.parent/"worker/slurm-session.json", repo)
        require(result["worker_terminal"] == reference["original_observer_interruption"]["worker_terminal"]
                and result["worker_terminal"]["terminal"] is True
                and result["worker_terminal"]["state"] == "FAILED"
                and result["worker_terminal"]["exit_code"] == "3:0"
                and result["session_id"] == ack["session_id"] == SESSION
                and str(result["job_id"]) == str(ack["job_id"]) == JOB,
                "historical_worker_terminal_binding_changed")
        require(manifest["method_status"] == "failed" and manifest["generation_complete"] is True
                and candidate["source_hash"] == SOURCE and candidate["artifact_hash"] == ARTIFACT,
                "historical_method_or_artifact_changed")
        require(progress == {"iteration": 3, "model_calls": 2, "attempt_count": 4, "error": None}
                and config["max_trials"] == 3 and config["run_id"] == TASK, "public_progress_changed")
        failed = [row for row in observer["observations"].values() if row.get("evidence_id") == "original-attempt-3"]
        require(len(failed) == 1 and failed[0]["status"] == "metric_unavailable", "observer_failure_changed")
        error = manifest["method_error"]
        require(isinstance(error, str) and error.startswith("ObserverInfrastructureError:")
                and "[Errno 122] Disk quota exceeded:" in error
                and "/sessions/"+SESSION+"/requests/observer-" in error, "observer_quota_cause_changed")
        trace = raw(generation/"error.log", repo).decode()
        require("observe_attempt(state)" in trace and "observer.observe(" in trace
                and "[Errno 122] Disk quota exceeded:" in trace, "observer_failure_stack_changed")
        snapshot_path, source_pin = authenticate_original(repo, generation, manifest)
        sys.path.insert(0, str(repo))
        from optimizer_agent import upstream_original as bridge
        from optimizer_agent.suite_orchestrator import load_candidate
        require(Path(candidate["source_path"]) == generation/"frozen-original.cu", "retained_source_path_changed")
        require(hashlib.sha256(raw(generation/"frozen-original.cu", repo)).hexdigest() == SOURCE,
                "retained_source_bytes_changed")
        require(load_candidate(candidate).artifact_hash == ARTIFACT, "retained_launch_artifact_changed")
        cli = closed_cli_receipts(generation, bridge, manifest)
        # Internal deserializer/import warnings may contain repr(state). Keep
        # them inside this trusted process; failures expose only a static code.
        sink = io.StringIO()
        previous_logging = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                projection = checkpoint_projection(generation, snapshot_path, config, history, bridge)
        finally:
            logging.disable(previous_logging)
        for item in checked:
            require(item == proof(repo/item["path"], repo), "public_evidence_drifted")
        require(public_task(study) == task, "public_task_drifted")
    return {"schema_version": 1, "kind": "read_only_original_terminal_phase_assessment",
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": task, "upstream": source_pin, "completed_cli_receipts": cli,
        "historical_method_status": "failed", "historical_observer_status": "metric_unavailable",
        "failure_phase": "external_observer_after_evaluate_candidate_iteration_3",
        "observer_errno": 122, "selected_artifact_hash": ARTIFACT, "checkpoint_projection": projection,
        "public_evidence": checked, "model_calls": 0, "gpu_requests": 0, "scheduler_calls": 0,
        "checkpoint_nodes_executed": 0, "campaign_files_written": 0,
        "private_values_exposed": False, "automatic_recovery_authorized": False,
        "next_requirement": "A separate reviewed recovery transaction must preserve the failed method/observer history, complete only finalize_result/END in a cold copy, and independently authorize any new observer/export-validation epoch. This assessment neither runs nor approves it."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Read actual checkpoint only after primary review")
    parser.add_argument("--output", type=Path, help="Fresh public receipt strictly under work/live")
    args = parser.parse_args()
    if not args.inspect:
        print(json.dumps({"mode": "protocol_only", "checkpoint_opened": False,
            "review_required_before_inspection": True, "scope": __doc__}, indent=2))
        return 0
    require(args.output is not None, "fresh_output_required")
    output = safe(args.output.absolute(), REPO/"work/live")
    require(not output.exists(), "output_exists")
    try:
        result = assess()
    except Refused as error:
        print(json.dumps({"status": "refused", "code": str(error), "private_values_exposed": False}))
        return 2
    except Exception:
        print(json.dumps({"status": "refused", "code": "internal_inspection_error",
                          "private_values_exposed": False}))
        return 2
    body = (json.dumps(result, indent=2, allow_nan=False)+"\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "assessed_only", "receipt": str(output),
        "sha256": hashlib.sha256(body).hexdigest(), "checkpoint_projection": result["checkpoint_projection"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
