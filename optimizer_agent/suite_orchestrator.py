"""Durable §16 campaign scheduler and sealed, same-GPU final groups.

One controller owns the journal lock. A stable worker intent is committed
before submission; resumption attaches to that exact session. An ambiguous
or terminal incomplete attempt is retained, never silently regenerated.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import sqlite3
import subprocess
import time
import traceback

from .campaign_config import CampaignConfig
from .quality_suite import TOOLS, stable_id, summarize_suite
from .protocol_incidents import load_protocol_incidents
from .model_usage import aggregate_model_usage, inspect_model_usage

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "failed", "recovery_required"}


def _infrastructure_diagnostic_lines(message):
    # Never reinterpret provider/compiler text quoting an infrastructure example.
    if re.search(r"(?i)\b(?:codex|openai|model|nvcc|nvrtc|compilation|compiler)\b", message):
        return []
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    known_pq_banner = [
        "** WARNING: connection is not using a post-quantum key exchange algorithm.",
        '** This session may be vulnerable to "store now, decrypt later" attacks.',
        "** The server may need to be upgraded. See https://openssh.com/pq.html",
    ]
    while lines:
        if lines[:3] == known_pq_banner:
            del lines[:3]
        elif re.fullmatch(r"Warning: Permanently added '[^']+' \([A-Za-z0-9-]+\) to the list of known hosts\.", lines[0]):
            lines.pop(0)
        else:
            break
    return lines


def remote_storage_failure_directory(error):
    """Identify only a complete mkdir EDQUOT/ENOSPC worker-session diagnostic."""
    if type(error).__name__ == "CodexError" or type(error).__module__.endswith("codex_backend"):
        return None
    if isinstance(error, subprocess.CalledProcessError):
        command = error.cmd
        executable = command[0] if isinstance(command, (list, tuple)) and command else None
        if not executable or Path(str(executable)).name != "ssh" or not error.returncode:
            return None
        message = error.stderr or ""
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
    else:
        message = str(error)
    lines = _infrastructure_diagnostic_lines(message)
    if len(lines) != 1:
        return None
    diagnostic = re.sub(r"^BackendError:\s*", "", lines[0])
    match = re.fullmatch(
        r'''mkdir: cannot create directory (?:'(?P<single>[^'\x00\r\n]+)'|‘(?P<curly>[^’\x00\r\n]+)’|"(?P<double>[^"\x00\r\n]+)"): (?:Disk quota exceeded|No space left on device)''',
        diagnostic)
    if not match:
        return None
    path = next(value for value in match.groupdict().values() if value is not None)
    directory = PurePosixPath(path)
    if (not directory.is_absolute() or str(directory) != path or ".." in directory.parts
            or len(directory.parts) < 6 or directory.parts[-2] != "sessions"
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", directory.name) is None):
        return None
    return path


def infrastructure_category(error):
    """Recognize explicit transport/storage failures, never a numerical loss.

    A generic timeout can belong to the model or compiler. Only an SSH process
    or an explicit transport diagnostic qualifies for transport recovery.
    Storage admission additionally requires one complete worker mkdir error.
    """
    if remote_storage_failure_directory(error) is not None:
        return "remote_storage_failure"
    if type(error).__name__ == "CodexError" or type(error).__module__.endswith("codex_backend"):
        return None
    if isinstance(error, subprocess.TimeoutExpired):
        command = error.cmd
        executable = command[0] if isinstance(command, (list, tuple)) and command else None
        return "transport_timeout" if executable and Path(str(executable)).name == "ssh" else None
    if isinstance(error, subprocess.CalledProcessError):
        command = error.cmd
        executable = command[0] if isinstance(command, (list, tuple)) and command else None
        if not executable or Path(str(executable)).name != "ssh" or error.returncode != 255:
            return None
        message = error.stderr or ""
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
    else:
        message = str(error)
    if message.startswith(("Atomic SSH upload timed out", "Remote artifact read timed out", "Remote request timed out")):
        return "transport_timeout"
    if message.startswith("Worker session ") and " is terminal (job " in message:
        return "worker_terminal_incomplete"
    lines = _infrastructure_diagnostic_lines(message)
    diagnostic = lines[0] if lines else ""
    diagnostic = re.sub(r"^BackendError:\s*", "", diagnostic)
    timeout_patterns = (
        r"ssh: connect to host \S+ port \d+: (?:connection|operation) timed out\b",
        r"ssh_dispatch_run_fatal: connection to \S+ port \d+: operation timed out\b",
        r"timeout, server \S+ not responding\.?$",
        r"connection to \S+(?: port \d+)? timed out\b",
        r"ssh (?:command |connection |transport |request )?(?:timeout|timed out)\b",
    )
    connection_patterns = (
        r"ssh: could not resolve hostname \S+: ",
        r"ssh: connect to host \S+ port \d+: (?:can(?:not|'t) assign requested address|connection (?:refused|reset|aborted|closed|failure)|network is unreachable|no route to host)\b",
        r"ssh_dispatch_run_fatal: connection to \S+ port \d+: can(?:not|'t) assign requested address\b",
        r"(?:kex_exchange_identification|ssh_exchange_identification): (?:read: )?(?:connection (?:reset|closed)|banner exchange: connection to )",
        r"connection to \S+(?: port \d+)? closed by remote host\.?$",
        r"client_loop: send disconnect: broken pipe\b",
        r"ssh: (?:temporary failure in name resolution|connection failure)\b",
    )
    if any(re.match(pattern,diagnostic,re.IGNORECASE) for pattern in timeout_patterns):
        return "transport_timeout"
    if any(re.match(pattern,diagnostic,re.IGNORECASE) for pattern in connection_patterns):
        return "ssh_transport_failure"
    return None


def _local_session_job(task, backend):
    """Authenticate a local submission acknowledgement without network access.

    This only restores journal provenance. A missing in-memory job is never
    attached or made executable here; recovery authenticates the remote owner.
    """
    results_dir = getattr(backend,"results_dir",None)
    if results_dir is None or not (Path(results_dir)/"slurm-session.json").exists():
        return None
    record = read_json(Path(results_dir)/"slurm-session.json")
    if not isinstance(record,dict) or record.get("uncertain") or "job_ids" in record:
        raise ValueError("Local submission acknowledgement is ambiguous")
    job = record.get("job_id")
    if isinstance(job,bool) or not isinstance(job,(str,int)) or not re.fullmatch(r"[1-9][0-9]*",str(job)):
        raise ValueError("Local submission acknowledgement has invalid scheduler job ID")
    if (record.get("session_id") != task["session"] or getattr(backend,"session_id",None) != task["session"]
            or not isinstance(getattr(backend,"remote_dir",None),str) or not backend.remote_dir
            or record.get("remote_dir") != backend.remote_dir):
        raise ValueError("Local submission acknowledgement session/directory ownership mismatch")
    config = getattr(backend,"config",None)
    saved_config = dict(record.get("config", {})) if isinstance(record.get("config"), dict) else None
    # This controller-only liveness option did not exist in historical leases.
    # It cannot change GPU resources, inputs, kernels or measurement protocols.
    if saved_config is not None and "reconnect_transport" not in saved_config and hasattr(config, "reconnect_transport"):
        saved_config["reconnect_transport"] = config.reconnect_transport
    if not is_dataclass(config) or saved_config != json.loads(json.dumps(asdict(config))):
        raise ValueError("Local submission acknowledgement configuration ownership mismatch")
    current = getattr(backend,"job_id",None)
    if ((task.get("job_id") is not None and str(task["job_id"]) != str(job))
            or (current is not None and str(current) != str(job))):
        raise ValueError("Local submission acknowledgement and journal/backend job ownership mismatch")
    if getattr(backend,"historical_only",False) and current is None:
        raise ValueError("Failed historical attachment cannot be authenticated from a local acknowledgement alone")
    return str(job)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_native_preflight(directory, value):
    """Keep every old/new probe byte string before replacing the latest view."""
    directory = Path(directory)
    latest = directory / "native-read-isolation-preflight.json"
    archive = directory / "native-read-isolation-preflight-archive"
    if archive.is_symlink() or latest.is_symlink():
        raise ValueError("Native preflight receipt paths must not be symlinks")
    if latest.exists() and (not latest.is_file() or latest.stat().st_nlink != 1):
        raise ValueError("Native preflight latest must be a regular file without aliases")
    archive.mkdir(parents=True, exist_ok=True)

    def sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    sync_directory(directory)

    def preserve(raw):
        digest = hashlib.sha256(raw).hexdigest()
        destination = archive / (digest + ".json")
        if destination.is_symlink() or (destination.exists() and (
                not destination.is_file() or destination.stat().st_nlink != 1)):
            raise ValueError("Native preflight archive must be a regular file without aliases")
        try:
            with destination.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if destination.read_bytes() != raw:
                raise ValueError("Native preflight archive content-address mismatch")
        sync_directory(archive)
        return {"path": str(destination), "sha256": digest}

    previous = preserve(latest.read_bytes()) if latest.exists() else None
    raw = json.dumps(value, indent=2, allow_nan=False, default=str).encode()
    current = preserve(raw)
    write_json(latest, value)
    sync_directory(directory)
    return {"previous": previous, "current": current, "latest": str(latest)}


def config_from_dict(data):
    data = dict(data)
    for name in ("actions", "search_sizes", "final_sizes", "required_sanitizers"):
        if name in data:
            data[name] = tuple(data[name])
    return CampaignConfig(**data)


def candidate_receipt(candidate):
    return {"candidate_id": candidate.candidate_id, "source_path": str(candidate.source),
            "source_hash": candidate._source_hash, "artifact_hash": candidate.artifact_hash,
            "kernel": candidate.kernel, "launch_rule": asdict(candidate.launch_rule),
            "parameters": dict(candidate.parameters)}


def load_candidate(receipt, *, destination=None):
    # Import the same direct modules as CampaignRuntime and the harness client.
    from .campaign import Candidate
    from .frozen_launch import parse_launch_rule
    path = Path(receipt.get("source_path", receipt.get("source", ""))).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt["source_hash"]:
        raise ValueError("Frozen candidate source hash mismatch")
    if destination:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("An existing frozen source cannot be replaced")
        destination.write_bytes(data)
        path = destination
    result = Candidate(receipt["candidate_id"], path, receipt["kernel"],
                       parse_launch_rule(receipt["launch_rule"]), receipt.get("parameters", {}))
    if result.artifact_hash != receipt["artifact_hash"]:
        raise ValueError("Frozen candidate artifact hash mismatch")
    return result


class SuiteJournal:
    def __init__(self, directory, plan):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "suite.sqlite"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,
                    state TEXT NOT NULL,session TEXT NOT NULL,job_id TEXT,result TEXT,error TEXT,started REAL,finished REAL);
                CREATE TABLE IF NOT EXISTS sealed_groups(id TEXT PRIMARY KEY,seed TEXT NOT NULL,commitment TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_attempts(task_id TEXT NOT NULL,attempt INTEGER NOT NULL,
                    snapshot TEXT NOT NULL,seed TEXT,commitment TEXT,reason TEXT NOT NULL,created REAL NOT NULL,
                    PRIMARY KEY(task_id,attempt));
            """)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(tasks)")}
            if "attempt" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0")
            previous = db.execute("SELECT value FROM metadata WHERE key='plan_hash'").fetchone()
            plan_hash = _hash(plan)
            if previous and previous[0] != plan_hash:
                raise ValueError("Suite plan changed; immutable campaign identities cannot be mixed")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('plan_hash',?)", (plan_hash,))
            db.execute("INSERT OR IGNORE INTO metadata VALUES('instance_id',?)", (secrets.token_hex(16),))
            instance = db.execute("SELECT value FROM metadata WHERE key='instance_id'").fetchone()[0]
            for entry in plan["entries"]:
                key = entry["config"]["run_id"]
                session = "suite-gen-" + hashlib.sha256((instance + ":" + key).encode()).hexdigest()[:32]
                db.execute("INSERT OR IGNORE INTO tasks(id,kind,payload,state,session) VALUES(?,?,?,?,?)",
                           (key, "generation", json.dumps(entry, sort_keys=True), "pending", session))
            for group in sorted({e["group_id"] for e in plan["entries"]}):
                entries = [e for e in plan["entries"] if e["group_id"] == group]
                # Private seed precommitted before any generator starts; final
                # data itself is instantiated only after every method is frozen.
                seed = secrets.randbits(63)
                db.execute("INSERT OR IGNORE INTO sealed_groups VALUES(?,?,?)",
                           (group, str(seed), hashlib.sha256(str(seed).encode()).hexdigest()))
                session = "suite-final-" + hashlib.sha256((instance + ":" + group).encode()).hexdigest()[:32]
                db.execute("INSERT OR IGNORE INTO tasks(id,kind,payload,state,session) VALUES(?,?,?,?,?)",
                           (group, "final", json.dumps(entries, sort_keys=True), "pending", session))
        os.chmod(self.path, 0o600)
        write_json(self.directory / "plan.json", plan)
        write_json(self.directory / "final-seed-commitments.json", self.commitments())

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @contextmanager
    def controller(self):
        with (self.directory / "controller.lock").open("a+") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another suite controller owns this journal") from error
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def tasks(self, kind=None):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM tasks" + (" WHERE kind=?" if kind else "") + " ORDER BY id", (kind,) if kind else ()).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"]), "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def task(self, key):
        with self.connection() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (key,)).fetchone()
        return ({**dict(row), "payload": json.loads(row["payload"]),
                 "result": json.loads(row["result"]) if row["result"] else None} if row else None)

    def update(self, key, **values):
        allowed = {"state", "job_id", "result", "error", "started", "finished", "session", "attempt"}
        if not values or set(values) - allowed:
            raise ValueError("Invalid journal update")
        if "result" in values:
            values["result"] = json.dumps(values["result"], allow_nan=False)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE tasks SET " + ",".join(f"{name}=?" for name in values) + " WHERE id=?", (*values.values(), key))

    def commitments(self):
        with self.connection() as db:
            return {r["id"]: r["commitment"] for r in db.execute("SELECT id,commitment FROM sealed_groups")}

    def seed(self, key):
        with self.connection() as db:
            return int(db.execute("SELECT seed FROM sealed_groups WHERE id=?", (key,)).fetchone()[0])


def _script(name):
    spec = importlib.util.spec_from_file_location("suite_" + name, REPO / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation_adapter(entry, runner, directory, session, *, resume=False, sensitive_root=None):
    config = config_from_dict(entry["config"])
    mode = entry["mode"]
    if mode == "true_upstream":
        from .upstream_original import run_upstream_campaign
        result = run_upstream_campaign(config, runner, directory, resume=resume, finalize=False,
                                       observe_first_gain=True)
    elif mode == "native_codex_vanilla":
        from .codex_backend import default_codex_executable
        from .vanilla_gpu_service import VanillaGPUService
        receipt_path = Path(directory) / "evidence/receipt.json"
        if receipt_path.exists():
            result = read_json(receipt_path)
            if result.get("status") != "completed":
                raise RuntimeError("Native one-prompt attempt is incomplete; preserve it rather than issue a duplicate prompt")
            result["candidate"] = read_json(Path(directory) / "evidence/candidate.json")
            observer_path = Path(directory) / "observer-metrics.json"
            if observer_path.exists():
                result["metrics"] = {**result.get("metrics", {}), **read_json(observer_path)}
            elif (Path(directory) / "evidence/gain-observer").is_dir():
                from .gain_observer import read_observer_metrics
                metrics = read_observer_metrics(Path(directory) / "evidence/gain-observer")
                write_json(observer_path, metrics)
                result["metrics"] = {**result.get("metrics", {}), **metrics}
        else:
            if Path(directory).exists() and any(Path(directory).iterdir()):
                raise RuntimeError("Native attempt has partial artifacts; an uncertain prompt is never automatically repeated")
            if sensitive_root is None:
                raise ValueError("A fresh native generator requires an explicit protected study root")
            services = []
            observer_started = time.time()
            def service_factory(config, protected, workspace, session_id, repo_dir):
                service = VanillaGPUService(config, protected, workspace, session_id, repo_dir=repo_dir,
                    runner=runner, observe_first_gain=True, observer_started_at_unix=observer_started)
                services.append(service)
                return service
            result = _script("run_codex_vanilla").run_vanilla(config, directory, session, default_codex_executable(),
                service_factory=service_factory, sensitive_root=Path(sensitive_root))
            if services:
                metrics = services[-1].observer_metrics()
                write_json(Path(directory) / "observer-metrics.json", metrics)
                result["metrics"] = {**result.get("metrics", {}), **metrics}
    else:
        from .campaign import run_on_runner
        result = run_on_runner(config, runner, Path(directory), resume=resume, generation_only=True)
    if not result.get("candidate"):
        launch = read_json(Path(directory) / "best-launch.json")
        source = Path(directory) / "best.cu"
        receipt = {**launch, "source_path": str(source.resolve()), "source_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "kernel": config.workload}
        result["candidate"] = candidate_receipt(load_candidate(receipt))
    candidate = load_candidate(result["candidate"])
    raw_usage = result.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    usage_observation = inspect_model_usage(raw_usage)
    declared_incomplete = (isinstance(raw_usage, dict) and
        (raw_usage.get("usage_complete") is False or bool(raw_usage.get("uncertain_model_intents"))
         or bool(raw_usage.get("uncertain_model_records"))))
    if declared_incomplete:
        # Legacy adapters may have initialized missing counters to zero. Those
        # values can only describe a partial observation, never a total for an
        # interrupted invocation. Preserve the supplied values separately.
        for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
            usage.setdefault("known_" + key, usage_observation.get("known_" + key, usage_observation.get(key)))
            usage[key] = None
            usage_observation[key] = None
        usage.setdefault("known_model_tokens", usage_observation["known_model_tokens"])
        usage_observation.update(model_tokens=None, usage_complete=False, declared_incomplete=True,
            unknown_fields=["input_tokens", "output_tokens"])
        usage["usage_complete"] = False
        known_calls = usage.get("known_model_calls")
        if known_calls is not None and (type(known_calls) is not int or known_calls < 0):
            usage["known_model_calls"] = None
        if (raw_usage.get("uncertain_model_intents") or raw_usage.get("uncertain_model_records")
                or usage.get("model_calls") == 0):
            # The legacy total can include unacknowledged intentions. Its raw
            # value is retained below, but is not a proven CLI invocation count.
            usage["model_calls"] = None
    usage["model_tokens"] = usage_observation["model_tokens"]
    if mode == "native_codex_vanilla":
        usage["model_calls"] = 1
        usage["native_completed_turns"] = len(result.get("usage_events", []))
        result.setdefault("elapsed_seconds", result.get("ended_at", time.time())-result["started_at"])
    # A method may fail after having produced a valid retained best. Preserve
    # that failure while allowing a transparent independent measurement of the
    # frozen artifact. Finishing the attempt does not make the method succeed.
    method_status = result.get("method_status")
    if method_status is None:
        method_status = "failed" if result.get("status") in {"failed", "method_failed", "generation_failed"} else "completed"
    if method_status not in {"completed", "failed"}:
        raise ValueError("A concluded generation requires an explicit completed or failed method status")
    result = {**result, "candidate": candidate_receipt(candidate), "generation_complete": True,
              "mode": mode, "status": "method_failed" if method_status == "failed" else "generation_complete",
              "method_status": method_status, "generation_stage_status": "generation_complete",
              "evidence_path": str(Path(directory).resolve()),
              "usage": usage, "usage_observation": usage_observation, "generation_usage_reported": raw_usage,
              "model_call_unit":"Codex CLI invocations; provider-internal model/tool cycles may be unreported",
              "generation_scores_excluded_from_suite_final": True}
    return result


def observed_metrics(directory, result):
    """Report only recorded costs; absence is unknown, never a zero cost."""
    directory = Path(directory)
    trials = []
    for path in sorted((directory / "generation/trials").glob("*/result.json")):
        row = read_json(path)
        tell = row.get("tell", {})
        trials.append({"trial_id": path.parent.name, "valid": tell.get("valid"),
                       "evaluation_id": row.get("evaluation_id", tell.get("evaluation_id"))})
    configurations, configuration_sources, legacy_snapshots, unfinished_requests = [], [], [], []
    canonical_measurements = []
    # Canonical request receipts retain every repair. Compatibility aliases must
    # never be counted again, nor presented as complete historical coverage.
    from .campaign import evaluation_record
    for trial in sorted((directory / "generation/trials").glob("*")):
        for path in sorted((trial / "evaluation-requests").glob("*/request.json")):
            if not (path.parent / "completion.json").exists():
                unfinished_requests.append({"request_id": path.parent.name,
                    "path": str(path.relative_to(directory)), "scope": "unfinished; configuration history may be incomplete"})
        for path in sorted((trial / "evaluation-requests").glob("*/completion.json")):
            _, completion, measurement = evaluation_record(path.parent)
            canonical_measurements.append(measurement)
            tuning = path.parent / "autotuning.json"
            if "autotuning.json" in completion["files"]:
                row = read_json(tuning)
                configurations.extend(row["configurations"])
                configuration_sources.append({"path": str(tuning.relative_to(directory)),
                    "request_id": path.parent.name, "attempt_id": completion["attempt_id"],
                    "evaluation_id": completion["evaluation_id"], "observations": len(row["configurations"]),
                    "scope": "immutable_completed_evaluation_request"})
        path = trial / "autotuning.json"
        if path.exists():
            row = read_json(path)
            if not row.get("receipt_alias"):
                observations = row.get("configurations", row.get("trials", row.get("observations", [])))
                configurations.extend(observations)
                legacy_snapshots.append({"path": str(path.relative_to(directory)), "observations": len(observations),
                    "scope": "legacy_latest_snapshot_only; earlier overwritten repairs are not recoverable here"})
    evaluated = {}
    for path in sorted((directory / "generation/trials").glob("*/evaluation-*.json")):
        row = read_json(path)
        if row.get("receipt_alias"):
            continue
        identity = row.get("artifact_hash", row.get("candidate_id", row.get("evaluation_id")))
        if identity:
            evaluated[identity] = {"artifact_or_candidate_id": identity, "valid": row.get("ok"), "status": row.get("status")}
    for row in canonical_measurements:
        identity = row.get("artifact_hash", row.get("candidate_id", row.get("evaluation_id")))
        if identity:
            evaluated[identity] = {"artifact_or_candidate_id": identity, "valid": row.get("ok"), "status": row.get("status")}
    for row in configurations:
        if row.get("validation"):
            validation = row["validation"]
            identity = validation.get("artifact_hash", row.get("candidate_id"))
            evaluated[identity] = {"artifact_or_candidate_id": identity, "valid": validation.get("ok"), "status": validation.get("status")}
    native_responses = [read_json(path) for path in (directory / "generation/evidence/requests").glob("*/response.json")]
    native_configurations = {}
    for response in native_responses:
        if response.get("operation") in {"validate", "benchmark"} and response.get("artifact_hash"):
            identity = response["artifact_hash"]
            native_configurations[identity] = response
            evaluated[identity] = {"artifact_or_candidate_id":identity,"valid":response.get("ok"),
                                  "status":response.get("result",{}).get("status")}
    requests = []
    for path in sorted((directory / "request-receipts").rglob("*.json")):
        requests.append(read_json(path))
    reports = [row.get("report", {}) for row in requests]
    known_validity = [row for row in evaluated.values() if type(row["valid"]) is bool]
    events_path = directory / "generation/events.json"
    events = read_json(events_path) if events_path.exists() else []
    research_events = [row for row in events if "research" in str(row.get("kind", row.get("event", "")))]
    model_events = {row["payload"]["call_id"]: row for row in events if row.get("kind") == "model"}
    promotions = [row["created_at"] for row in events if row.get("kind") == "promotion"]
    clock_path = directory / "generation/clock.json"
    started = read_json(clock_path)["started_at"] if clock_path.exists() else None
    first = min(promotions) if promotions else None
    first_models = [row for row in model_events.values() if first is not None and row["created_at"] <= first]
    research_models = [row for row in model_events.values()
                       if isinstance(row["payload"].get("purpose"), str)
                       and "research" in row["payload"]["purpose"]]
    unknown_purpose_models = [row for row in model_events.values()
                              if not isinstance(row["payload"].get("purpose"), str)
                              or not row["payload"]["purpose"].strip()]
    # One successful call with unreported usage may have both model and
    # model_uncertain events. Preserve the uncertainty without counting twice.
    model_reservations = {row["payload"].get("reservation_id") for row in model_events.values()
                          if row["payload"].get("reservation_id")}
    unmatched_uncertain = {}
    for index, row in enumerate(events):
        if row.get("kind") != "model_uncertain":
            continue
        payload = row.get("payload", {})
        call_id, reservation_id = payload.get("call_id"), payload.get("reservation_id")
        if (call_id and call_id in model_events) or (reservation_id and reservation_id in model_reservations):
            continue
        identity = ("call", call_id) if call_id else ("reservation", reservation_id) if reservation_id else ("event", index)
        unmatched_uncertain[identity] = row
    uncertain_research = [row for row in unmatched_uncertain.values()
                          if not isinstance(row.get("payload", {}).get("purpose"), str)
                          or not row["payload"]["purpose"].strip()
                          or "research" in row["payload"]["purpose"]]
    uncertain_first = [row for row in unmatched_uncertain.values()
                       if first is not None and row.get("created_at", float("-inf")) <= first]
    research_usage = aggregate_model_usage((row["payload"].get("usage") for row in research_models),
                                           unreported_calls=len(uncertain_research) + len(unknown_purpose_models)) if model_events or unmatched_uncertain else None
    first_usage = aggregate_model_usage((row["payload"].get("usage") for row in first_models),
                                        unreported_calls=len(uncertain_first)) if first is not None and (first_models or uncertain_first) else None
    def recorded_duration(value):
        return value if (type(value) is int and value >= 0
                         or type(value) is float and value >= 0 and math.isfinite(value)) else None
    durations = [recorded_duration(row["payload"].get("elapsed_seconds")) for row in research_models]
    known_durations = [value for value in durations if value is not None]
    duration_sum_unknown = False
    try:
        known_research_wall = sum(known_durations) if (known_durations or research_usage is not None
            and research_usage["observed_calls"] == 0) else None
    except OverflowError:
        known_research_wall, duration_sum_unknown = None, True
    duration_sum_unknown = duration_sum_unknown or (
        known_research_wall is not None and recorded_duration(known_research_wall) is None)
    if duration_sum_unknown:
        known_research_wall = None
    unknown_research_durations = (len(durations) - len(known_durations)
                                  + len(uncertain_research) + len(unknown_purpose_models))
    native_audit_path = directory / "generation/evidence/requests.jsonl"
    native_audit = [json.loads(line) for line in native_audit_path.read_text().splitlines() if line.strip()] if native_audit_path.exists() else []
    first_provisional = next((row["timestamp"] for row in native_audit if row.get("operation") == "benchmark" and row.get("ok") is True), None)
    metrics = {"candidate_configurations_with_recorded_validity": len(known_validity),
        "correct_candidates": sum(row["valid"] for row in known_validity),
        "correct_candidate_fraction": (sum(row["valid"] for row in known_validity)/len(known_validity)) if known_validity else None,
        "trial_results": trials, "candidate_validation_results": list(evaluated.values()),
        "configuration_observations": len(configurations) + len(native_configurations),
        "configuration_observation_sources": configuration_sources,
        "legacy_configuration_snapshots": legacy_snapshots,
        "unfinished_evaluation_requests": unfinished_requests,
        "configuration_history_scope": "Canonical completed requests counted once; legacy snapshots cannot establish complete repair history",
        "native_gpu_service_requests": len(native_audit) if native_audit_path.exists() else None,
        "native_rejected_service_requests": sum(row.get("status") != "complete" for row in native_audit) if native_audit_path.exists() else None,
        "first_valid_search_measurement_unix": first_provisional,
        "first_valid_search_measurement_scope": "Provisional correctness-admitted search result; no paired useful-gain decision",
        "configurations_filtered_before_compilation": sum(row.get("status") == "filtered" for row in configurations),
        "configurations_independently_validated": sum(row.get("validation") is not None for row in configurations),
        "harness_requests": len(requests),
        "cuda_event_kernel_seconds": sum(sum(row.get("latency_samples_ms", []))/1000 for row in reports) if requests else None,
        "native_compile_stage_wall_seconds": sum((row.get("compile_ms") or 0)/1000 for row in reports) if requests else None,
        "native_reservation_wall_seconds": sum(row.get("reservation_wall_seconds", 0) for row in reports) if requests else None,
        "compile_cache_hits": sum(row.get("compile_cache_hit") is True for row in reports),
        "compile_cache_misses": sum(row.get("compile_cache_hit") is False for row in reports),
        "research_events": len(research_events) if events_path.exists() else None,
        "research_model_wall_seconds": known_research_wall if research_usage is not None and not unknown_research_durations and not duration_sum_unknown else None,
        "research_model_wall_seconds_known_lower_bound": known_research_wall,
        "research_model_wall_seconds_unknown_observations": unknown_research_durations,
        "research_model_wall_seconds_aggregation_unknown": duration_sum_unknown,
        "research_model_tokens": research_usage["model_tokens"] if research_usage is not None else None,
        "research_model_usage": research_usage,
        "research_model_usage_scope": "Recorded research-purpose calls only; completed or uncertain calls with unknown purpose conservatively prevent complete attribution, without assigning their known tokens or duration to research. Durations are sums of recorded call observations, not documentary end-to-end elapsed time or provider HTTP requests.",
        "model_observations_with_unknown_purpose": len(unknown_purpose_models),
        "unmatched_model_uncertain_observations": len(unmatched_uncertain),
        "time_to_first_useful_gain_seconds": first-started if first is not None and started is not None else result.get("time_to_first_useful_gain_seconds"),
        "model_tokens_to_first_useful_gain": first_usage["model_tokens"] if first_usage is not None else None,
        "model_usage_to_first_useful_gain": first_usage,
        "model_calls_to_first_useful_gain": len(first_models) if first_usage is not None and not uncertain_first else None,
        "known_model_calls_to_first_useful_gain": len(first_models) if first_usage is not None else None,
        "cuda_event_seconds_to_first_useful_gain": sum(sum(row.get("report", {}).get("latency_samples_ms", []))/1000 for row in requests if first and row.get("ended_at_unix", float("inf")) <= first) if first else None,
        "time_to_final_generation_seconds": result.get("elapsed_seconds"),
        "first_useful_gain_status": "observed" if first is not None else "no_promoted_gain" if events_path.exists() else "not_recorded_by_method",
        "gpu_seconds_ledger_scope": "Trial elapsed ledger includes model/research/SSH and can span interruption downtime; actual allocated GPU time requires distinct Slurm accounting receipts"}
    metrics["unknown_metrics"] = [name for name in (
        "time_to_first_useful_gain_seconds", "research_model_wall_seconds")
        if metrics[name] is None]
    metrics["unknown_token_metrics"] = [name for name in (
        "research_model_tokens", "model_tokens_to_first_useful_gain") if metrics[name] is None]
    return metrics


class RecordingRunner:
    def __init__(self, runner, directory):
        self.runner, self.directory = runner, Path(directory)

    def __getattr__(self, key):
        return getattr(self.runner, key)

    def run(self, run_id, request):
        receipt = {"run_id": run_id, "request": asdict(request), "started_at_unix": time.time()}
        try:
            receipt["report"] = self.runner.run(run_id, request)
            return receipt["report"]
        except Exception as error:
            receipt["error"] = str(error)
            raise
        finally:
            receipt["ended_at_unix"] = time.time()
            write_json(self.directory / (run_id + ".json"), receipt)


def evaluate_group(entries, generations, runner, directory, seed, commitment, *,
                   artifact_qualification_incidents=None, artifact_incident_observations=None):
    """Freeze all methods, then reveal fresh data once on one borrowed GPU."""
    from .campaign import WorkloadSuite, WorkloadContract, Evaluator, device_identity
    group_started = time.monotonic()

    def duration(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    class FinalCostRunner:
        """Observe existing RPCs, without retaining requests, buffers or outputs."""
        def __init__(self, delegate):
            self.delegate, self.observations = delegate, []
            self.stage, self.owner = "setup", None

        def __getattr__(self, name):
            if name == "run_sanitizer":
                getattr(self.delegate, name)  # Preserve capability detection.
                return lambda run_id, request, *args, **kwargs: self.observe(name, run_id, request, *args, **kwargs)
            return getattr(self.delegate, name)

        def observe(self, operation, run_id, request, *args, **kwargs):
            started = time.monotonic()
            item = {"request_id": run_id if isinstance(run_id, str) and run_id else None,
                    "operation": operation, "stage": self.stage, "candidate_mode": self.owner}
            try:
                report = getattr(self.delegate, operation)(run_id, request, *args, **kwargs)
                if isinstance(report, dict):
                    fingerprint = report.get("request_fingerprint")
                    item.update(request_fingerprint=fingerprint if isinstance(fingerprint, str)
                                and re.fullmatch(r"[0-9a-f]{64}", fingerprint) else None,
                                ok=report.get("ok") if type(report.get("ok")) is bool else None,
                                reported_native_reservation_seconds=duration(report.get("reservation_wall_seconds")),
                                reported_compile_seconds=(duration(report.get("compile_ms")) / 1000
                                    if duration(report.get("compile_ms")) is not None else None))
                return report
            except Exception as error:
                item["exception_type"] = type(error).__name__
                raise
            finally:
                item["rpc_wall_seconds"] = time.monotonic() - started
                self.observations.append(item)

        def run(self, run_id, request):
            return self.observe("run", run_id, request)

        def costs(self):
            by_id = {}
            for item in self.observations:
                if item["request_id"] is not None:
                    by_id.setdefault(item["request_id"], []).append(item)
            # A physical request observed in more than one candidate/stage is
            # shared; it never contributes to several candidate subtotals.
            owners = {key: (items[0]["candidate_mode"] if all(
                item["candidate_mode"] == items[0]["candidate_mode"] and item["stage"] == items[0]["stage"]
                for item in items) else None) for key, items in by_id.items()}

            def subtotal(owner):
                keys = sorted(key for key in by_id if owners[key] == owner)
                items = [item for key in keys for item in by_id[key]]
                unidentified = [item for item in self.observations if item["request_id"] is None] if owner is None else []
                items += unidentified  # Unidentifiable RPCs stay unallocated; never deduplicate their native costs.
                totals, missing, conflicts = {}, set(), set()
                for field in ("reported_native_reservation_seconds", "reported_compile_seconds"):
                    values = []
                    for key in keys:
                        observations = by_id[key]
                        fingerprints = {item.get("request_fingerprint") for item in observations
                                        if item.get("request_fingerprint") is not None}
                        observed = {item.get(field) for item in observations if item.get(field) is not None}
                        if len(fingerprints) > 1 or len(observed) > 1:
                            conflicts.add(key)
                        elif observed:
                            values.append(next(iter(observed)))
                        else:
                            missing.add(key)
                    totals[field] = sum(values) if values else None
                return {"observed_rpc_calls": len(items), "unique_request_ids": len(keys),
                        "unidentified_rpc_calls": len(unidentified),
                        "request_ids": keys, "rpc_wall_seconds": sum(item["rpc_wall_seconds"] for item in items),
                        **totals, "native_duration_missing_request_ids": sorted(missing),
                        "native_duration_conflict_request_ids": sorted(conflicts)}
            return {"shared": subtotal(None), "candidates": {entry["mode"]: subtotal(entry["mode"]) for entry in entries},
                    "unidentified_rpc_calls": sum(item["request_id"] is None for item in self.observations)}

    runner = FinalCostRunner(runner)
    directory = Path(directory)
    config = config_from_dict(entries[0]["config"])
    config = replace(config, run_id=entries[0]["group_id"], seed=seed, final_sizes=config.search_sizes)
    properties = runner.get_device_properties(0)
    suite = WorkloadSuite(config.workload, directory / "sealed-inputs", config)
    baseline = suite.baseline(REPO / config.baseline_source)
    candidates, results = {"baseline": baseline}, {}
    # Authenticate/copy artifacts before creating any held-out inputs.
    for entry in entries:
        generation = generations.get(entry["config"]["run_id"], {})
        try:
            receipt = generation["candidate"]
            candidates[entry["mode"]] = load_candidate(receipt, destination=directory / "frozen" / entry["mode"] / "kernel.cu")
        except Exception as error:
            results[entry["mode"]] = {"status": "generation_unavailable", "qualified": False, "error": str(error)}
    write_json(directory / "frozen-candidates.json", {key: candidate_receipt(value) for key, value in candidates.items()})
    available_modes = set(candidates)
    # A reviewed negative finding is independent of numerical/sanitizer success.
    # Preserve the frozen executable and historical generation, but never run or
    # rank a known-disqualified artifact as though racecheck covered global WAW.
    artifact_qualification_incidents = artifact_qualification_incidents or {}
    artifact_incident_observations = artifact_incident_observations or []
    global_incident_uncertainty = any(row.get("status") == "uncertain" and not row.get("artifact_hash")
                                      for row in artifact_incident_observations)
    uncertain_artifacts = {row["artifact_hash"] for row in artifact_incident_observations
                           if row.get("status") == "uncertain" and row.get("artifact_hash")}
    baseline_reasons = artifact_qualification_incidents.get(baseline.artifact_hash, [])
    if global_incident_uncertainty or baseline_reasons or baseline.artifact_hash in uncertain_artifacts:
        raise ValueError("FINAL qualification suspended: baseline or global artifact incident is not admitted")
    for mode, candidate in list(candidates.items()):
        if mode == "baseline":
            continue
        incidents = artifact_qualification_incidents.get(candidate.artifact_hash, [])
        uncertain = candidate.artifact_hash in uncertain_artifacts
        if incidents or uncertain:
            results[mode] = {**candidate_receipt(candidate), "qualified": False,
                "scientific_qualification_eligible": False,
                "status": "artifact_qualification_uncertain" if uncertain else "artifact_scientifically_disqualified",
                "artifact_qualification_incidents": incidents,
                "artifact_incident_observations": [row for row in artifact_incident_observations
                                                    if row.get("artifact_hash") == candidate.artifact_hash],
                "generation_method_status": generations.get(next(
                    entry["config"]["run_id"] for entry in entries if entry["mode"] == mode), {}).get("method_status"),
                "scope": "Exact artifact refused before FINAL input creation, validation or ranking. Historical SEARCH results and generation method status are unchanged."}
            del candidates[mode]
    cases = suite.cases(final=True)
    inputs = {str(Path(a.input_file).relative_to(directory)): hashlib.sha256(Path(a.input_file).read_bytes()).hexdigest()
              for case in cases for a in case.request.arguments if getattr(a, "input_file", None)}
    metadata = {"target_environment": device_identity(properties), "fixed_shape_domain": list(config.search_sizes),
        "seed": seed, "seed_commitment": commitment, "input_hashes": inputs,
        "scope": "Exclusive shared GPU; CUDA Event kernel time; input reset per launch; no transfer/compile timings in ranking",
        "generation_closed": True, "outcome_dependent_retries": False}
    from .workload_semantics import operation_spec
    metadata["operation_semantics"] = operation_spec(config.workload)
    write_json(directory / "holdout-reveal.json", metadata)
    contract = WorkloadContract(config.run_id, 1, config.workload, suite.signature,
        "Independent NumPy float64 CPU oracle with float32 outputs; fixed domain with fresh sealed data",
        benchmark_series=config.benchmark_series, launches_per_series=config.launches_per_series, warmup=config.warmup,
        confidence=config.confidence_level, min_relative_gain=config.min_relative_gain,
        max_critical_regression=config.max_critical_regression, confirmation_batches=config.confirmation_batches,
        required_sanitizers=TOOLS, allowed_precision="float32", metadata_json=json.dumps(metadata, sort_keys=True))
    evaluator = Evaluator(runner, contract, suite.oracle, {"short": cases[:1], "search": cases, "performance": cases})
    proof = {"schema_version": 1, "kind": "shared_final_group", "group_id": config.run_id,
        "workload": config.workload, "campaign_seed": entries[0]["seed"], "final_seed": seed, "seed_commitment": commitment,
        "contract_hash": evaluator.contract_hash, "contract": asdict(contract), "environment": device_identity(properties),
        "input_hashes": inputs, "shapes": list(config.search_sizes), "candidates": results,
        "baseline": candidate_receipt(baseline), "generation_scores_used": False, "execution_complete": False,
        "artifact_incident_observations": artifact_incident_observations,
        "artifact_qualification_excluded_modes": sorted(available_modes - candidates.keys()),
        "artifact_qualification_scope": "Reviewed negative findings refuse exact artifacts; no incident is not a proof of race freedom."}
    # All independent validations and sanitizers precede every ranking call.
    qualifications, qualification_memo = {}, {}
    for mode, candidate in candidates.items():
        # Scope this memo to this invocation, GPU, immutable cases and contract.
        # Every alias still checks source/launch/parameters, input bytes and
        # current hardware before referring to the original full evidence.
        context = evaluator.qualification_context(candidate, suite="search")
        previous = qualification_memo.get(context["key"]) if context["key"] else None
        if previous:
            original_mode, original = previous
            evidence_path = directory / "qualification" / (original_mode + ".json")
            evidence_bytes = evidence_path.read_bytes()
            if _hash(json.loads(evidence_bytes)) != _hash(original):
                raise ValueError("Qualification evidence changed before alias reuse")
            qualifications[mode] = {"evaluation_id": original["evaluation_id"], "candidate_id": candidate.candidate_id,
                "artifact_hash": candidate.artifact_hash, "contract_hash": evaluator.contract_hash, "suite": "search",
                "ok": original.get("ok") is True, "status": original["status"],
                "required_sanitizers": original.get("required_sanitizers", []), "qualification_context": context,
                "qualification_execution": "alias_of_same_group_evidence", "fresh_gpu_runs": 0,
                "ranking_eligible": False, "instrumented_timings_excluded": True,
                "qualification_evidence_ref": {"path": "qualification/" + original_mode + ".json",
                    "sha256": hashlib.sha256(evidence_bytes).hexdigest(), "digest_encoding": "raw file bytes",
                    "canonical_sha256": _hash(original),
                    "evaluation_id": original["evaluation_id"], "artifact_hash": original["artifact_hash"]}}
        else:
            runner.stage, runner.owner = "qualification", None if mode == "baseline" else mode
            qualifications[mode] = {**evaluator.qualify_sanitizers(candidate, suite="search"),
                "qualification_context": context, "qualification_execution": "fresh"}
            qualification = qualifications[mode]
            if (context["key"] and qualification.get("evaluation_id")
                    and qualification.get("artifact_hash") == candidate.artifact_hash
                    and qualification.get("contract_hash") == evaluator.contract_hash and qualification.get("suite") == "search"):
                qualification_memo[context["key"]] = mode, qualification
        write_json(directory / "qualification" / (mode + ".json"), qualifications[mode])
    transport_failures = [row for qualification in qualifications.values() for row in _walk(qualification)
                          if infrastructure_category(row.get("error", ""))]
    if transport_failures:
        proof.update(infrastructure_category=infrastructure_category(transport_failures[0]["error"]), infrastructure_interrupted=True,
                     qualifications=qualifications, evaluation_finished=False)
        write_json(directory / "group-result.pending.json", proof)
        return proof
    baseline_ok = qualifications["baseline"].get("ok") is True
    if baseline_ok:
        runner.stage, runner.owner = "baseline_measurement", None
        proof["baseline_measurement"] = evaluator.benchmark(baseline)
        baseline_ok = proof["baseline_measurement"].get("ok") is True
        write_json(directory / "group-result.pending.json", proof)
    for entry in entries:
        mode = entry["mode"]
        if mode not in candidates:
            continue
        candidate, qualification = candidates[mode], qualifications[mode]
        generation = generations.get(entry["config"]["run_id"], {})
        result = {**candidate_receipt(candidate), "qualified": qualification.get("ok") is True,
            "generation_method_status": generation.get("method_status"),
            "generation_method_error": generation.get("method_error"),
            "candidate_fallback": generation.get("candidate_fallback"),
            "sanitizer_qualification": qualification, "status": "final_validation_failed"}
        if result["qualified"] and baseline_ok:
            if candidate.artifact_hash == baseline.artifact_hash:
                baseline_context = qualifications["baseline"]["qualification_context"]
                candidate_context = qualification["qualification_context"]
                if baseline_context["key"] and baseline_context["key"] == candidate_context["key"]:
                    result["status"] = "validated_no_gain"
                    result["gain_basis"] = "exact_baseline_artifact"
                else:
                    result["status"] = "final_comparison_identity_unproven"
                    result["gain_basis"] = "unproven_artifact_context"
            else:
                runner.stage, runner.owner = "candidate_baseline_confirmation", mode
                confirmation = evaluator.confirm(candidate, baseline)
                result["confirmation"] = confirmation
                result["confirmation_completed_at_unix"] = time.time()
                generation = generations.get(entry["config"]["run_id"], {})
                generation_started = generation.get("task_started_at_unix")
                result["time_to_independently_confirmed_final_gain_seconds"] = (
                    result["confirmation_completed_at_unix"]-generation_started
                    if confirmation.get("status") == "promoted" and generation_started is not None else None)
                result["status"] = "certified" if confirmation.get("status") == "promoted" else "validated_gain_uncertified"
                result["gain_basis"] = "paired_measurement"
                if not confirmation.get("ok"):
                    result["status"] = "final_measurement_failed"
                    result["qualified"] = False
        elif not baseline_ok:
            result["status"] = "baseline_validation_failed"
            result["qualified"] = False
        proof["candidates"][mode] = result
        write_json(directory / "group-result.pending.json", proof)
    proof["baseline_qualification"] = qualifications["baseline"]
    proof["method_comparisons"] = {}
    # Predeclared direct pairs; one fixed confirmation each, including losses.
    for other in (entry["mode"] for entry in entries if entry["mode"] != "full"):
        if all(mode in candidates and qualifications[mode].get("ok") is True for mode in ("full", other)):
            runner.stage, runner.owner = "direct_method_comparison", None
            proof["method_comparisons"]["full_vs_" + other] = evaluator.confirm(candidates["full"], candidates[other])
            write_json(directory / "group-result.pending.json", proof)
    if all(mode in candidates and qualifications[mode].get("ok") is True for mode in ("pmpp_permanent_only", "no_pmpp")):
        runner.stage, runner.owner = "direct_method_comparison", None
        proof["method_comparisons"]["pmpp_permanent_only_vs_no_pmpp"] = evaluator.confirm(
            candidates["pmpp_permanent_only"], candidates["no_pmpp"])
        write_json(directory / "group-result.pending.json", proof)
    proof["all_modes_available"] = all(e["mode"] in available_modes for e in entries)
    transport_failures = [row for row in _walk(proof) if infrastructure_category(row.get("error", ""))]
    explicit_infrastructure_failure = any(row.get("status") == "infrastructure_failure" or row.get("worker_poisoned") is True
                                          for row in _walk(proof))
    if transport_failures or explicit_infrastructure_failure:
        proof.update(infrastructure_category=infrastructure_category(transport_failures[0]["error"]) if transport_failures
                     else "final_request_infrastructure_failure",
                     infrastructure_interrupted=True, evaluation_finished=False)
        write_json(directory / "group-result.pending.json", proof)
        return proof
    proof["evaluation_finished"] = True
    closed_at, group_wall = time.time(), time.monotonic() - group_started
    costs = runner.costs()
    cost_observation = {"schema_version": 1, "kind": "final_group_request_cost_observation",
        "group_id": config.run_id, "status": "closed", "closed_at_unix": closed_at,
        "contract_hash": evaluator.contract_hash, "environment": proof["environment"],
        "baseline_artifact_hash": baseline.artifact_hash,
        "candidate_artifact_hashes": {mode: candidate.artifact_hash for mode, candidate in candidates.items() if mode != "baseline"},
        "final_group_wall_seconds": group_wall, "request_costs": costs, "observations": runner.observations,
        "scope": "RPC wall time includes transport and waiting. Native durations are reported subtotals over unique worker request IDs; missing/conflicting values remain unknown. Shared baseline/direct-pair costs are not allocated to candidates. No Slurm allocation, model usage, USD or input buffers are inferred."}
    cost_path = directory / "final-cost-observation.json"
    write_json(cost_path, cost_observation)
    cost_evidence = {"path": "final-cost-observation.json", "sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest()}
    proof["final_cost_observation"] = {**cost_evidence, "shared_group_request_cost": costs["shared"]}
    for entry in entries:
        generation = generations.get(entry["config"]["run_id"], {})
        result = proof["candidates"][entry["mode"]]
        started = duration(generation.get("task_started_at_unix"))
        usage = generation.get("usage", generation.get("budgets", {}).get("spent", {}))
        usage = usage if isinstance(usage, dict) else {}
        result["final_closure"] = {"status": "closed", "outcome": result["status"], "group_id": config.run_id,
            "closed_at_unix": closed_at,
            "seconds_since_generation_task_start": closed_at - started if started is not None and started <= closed_at else None,
            "final_group_wall_seconds": group_wall,
            "reported_generation_usage": {key: usage.get(key) for key in ("model_tokens", "model_calls", "gpu_seconds")},
            "reported_generation_usage_source": "generation.usage" if "usage" in generation else
                "generation.budgets.spent" if "spent" in generation.get("budgets", {}) else "unreported",
            "candidate_request_cost": costs["candidates"][entry["mode"]],
            "shared_group_request_cost": costs["shared"], "cost_evidence": cost_evidence,
            "final_model_tokens": None, "final_model_calls": None, "cost_usd": None,
            "scope": "Final evaluation attempt closed after all group comparisons, independently of gain and numerical qualification; not a worker-terminal assertion or the exact instant of an individual candidate verdict. Elapsed since generation start includes queue/interruption. Generation ledger usage is a recorded, possibly partial snapshot; model usage at FINAL and USD are unknown. Candidate-specific confirmation includes both sides of its baseline comparison; other shared baseline/direct-pair costs are excluded."}
    return proof


class SuiteOrchestrator:
    def __init__(self, plan, directory, *, concurrency=2, backend_factory=None,
                 generate=generation_adapter, qualify=evaluate_group):
        if type(concurrency) is not int or not 2 <= concurrency <= 4:
            raise ValueError("GPU worker concurrency must be between two and four")
        self.plan, self.directory, self.concurrency = plan, Path(directory).resolve(), concurrency
        self.journal = SuiteJournal(self.directory, plan)
        self.backend_factory, self.generate, self.qualify = backend_factory, generate, qualify
        self._controller_epoch = None

    def _backend(self, task, config, directory):
        if self.backend_factory:
            return self.backend_factory(task, config, directory)
        from .campaign import SlurmBackend, SlurmConfig
        path = Path(config.execution_config)
        if not path.is_absolute():
            path = REPO / path
        return SlurmBackend(SlurmConfig.load(path), REPO, directory / "worker", session_id=task["session"])

    def _directory(self, task):
        root = self.directory / ("campaigns" if task["kind"] == "generation" else "final-groups") / task["id"]
        return root if task.get("attempt", 0) == 0 else root / "attempts" / ("%03d" % task["attempt"])

    def _require_active_study(self):
        if (self.directory / "superseded-by.json").exists():
            raise ValueError("This study is superseded; retain its receipts and use the replacement study directory")

    def _dispatch_key(self, task):
        """Prefer locally authenticated recovery work, without inferring queue state.

        These read-only hints never authorize execution. Remote ownership,
        replay contents and recovery guards remain checked by the worker.
        Missing or ambiguous local proof merely loses its extra priority.
        """
        def timestamp(*values):
            return next((float(value) for value in values if type(value) in (int, float)
                         and math.isfinite(value) and value >= 0), math.inf)

        def record(path):
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Dispatch proof must be an object")
            return value, raw

        def committed(value):
            return (value.get("schema_version") == 1 and value.get("phase") == "committed"
                    and value.get("task_id") == task["id"])

        directory = self._directory(task)
        attempt = task.get("attempt", 0)
        if task["kind"] == "generation":
            try:
                audit, _ = record(directory / "bootstrap-recovery.json")
                replay_path = directory / "bootstrap-replay-manifest.json"
                replay, replay_bytes = record(replay_path)
                ack, _ = record(directory / "worker" / "slurm-session.json")
                ack_config = ack.get("config", {})
                hardware = replay.get("hardware", {})
                root = ack_config.get("remote_root") if isinstance(ack_config, dict) else None
                if (task["state"] == "running" and re.fullmatch(r"[1-9][0-9]*", str(task.get("job_id")))
                        and committed(audit) and audit.get("same_live_allocation") is True
                        and audit.get("previous_session") == task["session"]
                        and str(audit.get("previous_job_id")) == str(task["job_id"])
                        and audit.get("replay_manifest") == str(replay_path.resolve())
                        and audit.get("replay_manifest_sha256") == hashlib.sha256(replay_bytes).hexdigest()
                        and replay.get("schema_version") == 1
                        and replay.get("kind") == "pre_model_bootstrap_prefix"
                        and replay.get("receipt_directory") == str((directory / "request-receipts").resolve())
                        and replay.get("manifest_hash") == _hash({k:v for k,v in replay.items() if k != "manifest_hash"})
                        and isinstance(hardware, dict)
                        and str(hardware.get("slurm_job_id")) == str(task["job_id"])
                        and replay.get("hardware_hash") == _hash(hardware)
                        and isinstance(replay.get("entries"), list) and replay["entries"]
                        and not ack.get("uncertain") and "job_ids" not in ack
                        and ack.get("session_id") == task["session"]
                        and str(ack.get("job_id")) == str(task["job_id"])
                        and isinstance(root, str) and root
                        and ack.get("remote_dir") == root.rstrip("/") + "/sessions/" + task["session"]):
                    return (0, timestamp(task.get("started"), audit.get("created_at_unix")), task["id"], attempt)
            except (OSError, ValueError, TypeError, KeyError):
                pass
            try:
                source_recovery = task["payload"].get("source_recovery", {})
                marker_path = directory / "generation" / "unscored-rehost.json"
                intent, intent_bytes = record(self.directory / "unscored-rehost-intents" / (task["id"] + ".json"))
                marker, marker_bytes = record(marker_path)
                context_id = source_recovery.get("context_id")
                if (source_recovery.get("kind") == "unscored_performance_context_transfer"
                        and source_recovery.get("fresh_research_required") is True
                        and isinstance(context_id, str) and context_id
                        and source_recovery.get("marker_path") == str(marker_path.resolve())
                        and committed(intent) and committed(marker) and intent_bytes == marker_bytes
                        and intent.get("kind") == source_recovery["kind"]
                        and intent.get("context_id") == context_id
                        and intent.get("new_session") == task["session"]
                        and intent.get("new_attempt") == attempt
                        and intent.get("marker_path") == str(marker_path.resolve())
                        and intent.get("new_generation") == str(marker_path.parent.resolve())):
                    return (1, timestamp(intent.get("created_at_unix")), task["id"], attempt)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
            if attempt:
                try:
                    audit, _ = record(self._directory({**task, "attempt": attempt-1}) / "bootstrap-recovery.json")
                    if (task["state"] == "pending" and committed(audit)
                            and audit.get("same_live_allocation") is False
                            and audit.get("new_session") == task["session"]
                            and audit.get("new_attempt") == attempt):
                        return (2, timestamp(audit.get("created_at_unix")), task["id"], attempt)
                except (OSError, ValueError, TypeError, KeyError):
                    pass
        return (2 if task["state"] == "running" else 3,
                timestamp(task.get("started")) if task["state"] == "running" else math.inf,
                task["id"], attempt)

    def _task(self, task):
        # Spawned workers can outlive a killed controller. Their own lock
        # prevents a resumed controller from dispatching the same generation
        # twice while the original model/GPU process is still alive.
        lock_dir = self.directory / "task-locks"
        lock_dir.mkdir(exist_ok=True)
        with (lock_dir / (hashlib.sha256(task["id"].encode()).hexdigest() + ".lock")).open("a+") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                # A future may wait while an older surviving worker finishes.
                # Use fresh state under the task lock, retaining the enqueued
                # immutable identity; never retry its newly recorded failure.
                fresh = self.journal.task(task["id"])
                if (fresh is None or fresh["state"] not in {"pending", "running"}
                        or any(fresh.get(key) != task.get(key) for key in ("kind", "session", "attempt", "payload"))):
                    return
                task = fresh
                rehost_intent = self.directory / "unscored-rehost-intents" / (task["id"]+".json")
                if rehost_intent.exists():
                    intent = read_json(rehost_intent)
                    if intent.get("phase") == "prepared" and task["session"] in {intent.get("previous_session"),intent.get("new_session")}:
                        return
                preparation_dirs = [self._directory(task)]
                if task.get("attempt",0):
                    preparation_dirs.append(self._directory({**task,"attempt":task["attempt"]-1}))
                for prepared_dir in preparation_dirs:
                    intent_path = prepared_dir / "bootstrap-recovery.json"
                    if intent_path.exists():
                        intent = read_json(intent_path)
                        if (intent.get("phase") == "prepared" and intent.get("task_id") == task["id"]
                                and task["session"] in {intent.get("previous_session"),intent.get("new_session")}):
                            # An interrupted local preparation must finish its
                            # own transaction before any scheduler/model work.
                            return
                from .evaluation_recovery import preparation_pending, validate_task_recovery
                if preparation_pending(self, task):
                    return
                validate_task_recovery(self, task)
                from .dispatch_attestation import capture_dispatch_attestation
                capture_dispatch_attestation(self.directory, task=task, controller_epoch=self._controller_epoch)
                self._execute_task(task)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def _commit_terminal_result(self, task, result, terminal, started, ready_at, *, owned_job_id):
        """Commit lifecycle metadata only; callers authenticate the owned worker."""
        result_path = self._directory(task) / "result.json"
        result["worker_terminal"] = terminal
        result["task_started_at_unix"] = started
        result["task_finished_at_unix"] = time.time()
        result["total_task_wall_seconds_including_queue"] = result["task_finished_at_unix"]-started
        result["queue_and_worker_start_wall_seconds"] = ready_at-started if ready_at is not None else None
        result["execution_complete"] = bool(terminal.get("terminal") and terminal.get("state") == "COMPLETED"
            and str(terminal.get("exit_code")) == "0:0" and str(terminal.get("job_id")) == str(owned_job_id))
        if task["kind"] == "generation":
            result["execution_complete"] &= result.get("generation_complete") is True
        else:
            result["execution_complete"] &= result.get("evaluation_finished") is True
        if not terminal.get("terminal") or terminal.get("state") != "COMPLETED" or str(terminal.get("exit_code")) != "0:0":
            result["infrastructure_category"] = "worker_terminal_incomplete"
        write_json(result_path, result)
        self.journal.update(task["id"], state="completed" if result["execution_complete"] else "recovery_required",
                            result=result, finished=time.time())

    def _execute_task(self, task):
        kind = task["kind"]
        entry = task["payload"] if kind == "generation" else task["payload"][0]
        config = config_from_dict(entry["config"])
        directory = self._directory(task)
        result_path = directory / "result.json"
        backend, result, terminal, attached_status = None, None, None, None
        started = task["started"] or time.time()
        ready_at = None
        self.journal.update(task["id"], state="running", started=started)
        try:
            # A result written before a controller interruption is authoritative;
            # recover the exact worker only to obtain its terminal receipt.
            if result_path.exists():
                result = read_json(result_path)
                if result.get("worker_terminal", {}).get("terminal"):
                    if result.get("candidate"):
                        load_candidate(result["candidate"])
                    terminal = result["worker_terminal"]
                    success = (result.get("execution_complete") is True and terminal.get("state") == "COMPLETED"
                        and str(terminal.get("exit_code")) == "0:0" and str(terminal.get("job_id")) == str(task["job_id"])
                        and result.get("session_id") == task["session"])
                    self.journal.update(task["id"], state="completed" if success else "recovery_required", result=result, finished=time.time())
                    return
            if kind == "final" and result is None and (directory / "holdout-reveal.json").exists():
                raise RuntimeError("Final inputs were already revealed before interruption; no outcome-dependent final repeat is permitted")
            native_completed = directory / "generation/evidence/receipt.json"
            if (result is None and kind == "generation" and entry["mode"] == "native_codex_vanilla"
                    and self.generate is generation_adapter and native_completed.exists()
                    and read_json(native_completed).get("status") == "completed"):
                # The CLI can finish before the outer controller commits its
                # task result. Recover those frozen files without a new prompt
                # or any GPU access, even if its old worker has since ended.
                result = self.generate(entry, None, directory / "generation", task["session"],
                                       resume=True, sensitive_root=self.directory.parent)
                result["metrics"] = {**observed_metrics(directory, result), **result.get("metrics", {})}
                result.update(session_id=task["session"], job_id=task["job_id"], task_evidence_path=str(directory))
                write_json(result_path, result)
            if (result is None and kind == "generation" and entry["mode"] == "native_codex_vanilla"
                    and self.generate is generation_adapter):
                from .codex_backend import default_codex_executable
                from .native_read_isolation import probe_read_isolation
                preflight = probe_read_isolation(self.directory.parent, codex_executable=default_codex_executable())
                write_native_preflight(directory, preflight)
                if preflight.get("ok") is not True:
                    raise ValueError("Native generator read-isolation preflight did not pass")
            backend = self._backend(task, config, directory)
            if result is not None:
                attached_status = backend.attach_existing()
                if task["job_id"] is not None and str(task["job_id"]) != str(backend.job_id):
                    backend.job_id = None
                    raise ValueError("Historical worker differs from the journal's owned job")
            else:
                backend.start()
            self.journal.update(task["id"], job_id=str(backend.job_id))
            if result is None:
                backend.wait_until_ready()
                ready_at = time.time()
                replay = directory / "bootstrap-replay-manifest.json"
                recording_dir = directory / "request-receipts"
                if replay.exists():
                    recording_dir /= "recovered"
                recording = RecordingRunner(backend.runner(), recording_dir)
                if replay.exists():
                    from .bootstrap_replay import BootstrapReplayRunner
                    recovery = read_json(directory / "bootstrap-recovery.json")
                    if (recovery.get("previous_session") != task["session"]
                            or str(recovery.get("previous_job_id")) != str(backend.job_id)
                            or hashlib.sha256(replay.read_bytes()).hexdigest() != recovery.get("replay_manifest_sha256")):
                        raise ValueError("Bootstrap replay must retain its authenticated allocation and manifest")
                    recording = BootstrapReplayRunner(recording, replay, backend.hardware,
                                                       directory / "bootstrap-replay-audit",
                                                       defer_request_files=entry["mode"] == "native_codex_vanilla")
                if kind == "generation":
                    kwargs = {"resume": task["state"] == "running" or bool(entry.get("source_recovery"))}
                    if self.generate is generation_adapter:
                        kwargs["sensitive_root"] = self.directory.parent
                    result = self.generate(task["payload"], recording, directory / "generation", task["session"], **kwargs)
                else:
                    generations = {r["id"]: r["result"] or {} for r in self.journal.tasks("generation")}
                    kwargs = {}
                    if self.qualify is evaluate_group:
                        from .artifact_qualification_incidents import load_artifact_qualification_incidents
                        incidents, observations = load_artifact_qualification_incidents(self.directory,
                            [entry["config"]["run_id"] for entry in self.plan["entries"]])
                        kwargs = {"artifact_qualification_incidents": incidents,
                                  "artifact_incident_observations": observations}
                    result = self.qualify(task["payload"], generations, recording, directory,
                        self.journal.seed(task["id"]), self.journal.commitments()[task["id"]], **kwargs)
                result["metrics"] = {**observed_metrics(directory, result), **result.get("metrics", {})}
                result.update(session_id=task["session"], job_id=str(backend.job_id), task_evidence_path=str(directory))
                write_json(result_path, result)
            result.update(session_id=task["session"], job_id=str(backend.job_id), task_evidence_path=str(directory))
            if not attached_status or attached_status.get("terminal") is not True:
                backend.close()
            terminal = backend.wait_until_finished(timeout=60)
            self._commit_terminal_result(task, result, terminal, started, ready_at, owned_job_id=backend.job_id)
        except Exception as error:
            owned_job, ownership_error, journal_job = None, None, task.get("job_id")
            if backend is not None:
                try:
                    current_task = next(row for row in self.journal.tasks(kind) if row["id"] == task["id"])
                    journal_job = current_task.get("job_id")
                    owned_job = _local_session_job(current_task,backend)
                except (ValueError,TypeError,OSError) as ownership_failure:
                    ownership_error = str(ownership_failure)
            if owned_job is not None:
                self.journal.update(task["id"],job_id=owned_job)
                journal_job = owned_job
            failure = {"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                "session_id": task["session"],
                "job_id": owned_job, "journal_job_id": journal_job,
                "execution_complete": False, "partial_result": result, "infrastructure_category": infrastructure_category(error)}
            if ownership_error:
                failure["job_ownership_error"] = ownership_error
            if backend is not None and getattr(backend, "job_id", None) and not ownership_error:
                try:
                    if not attached_status or attached_status.get("terminal") is not True:
                        backend.close()
                    failure["worker_terminal"] = backend.wait_until_finished(timeout=60)
                except Exception as close_error:
                    failure["cleanup_error"] = str(close_error)
            write_json(directory / "failure.json", failure)
            self.journal.update(task["id"], state="recovery_required", result=result or failure, error=str(error), finished=time.time())

    def recover_bootstrap(self, task_id, reason):
        from .bootstrap_recovery import recover_pre_model
        return recover_pre_model(self, task_id, reason)

    def resume_bootstrap_preparation(self, task_id):
        from .bootstrap_recovery import resume_preparation
        return resume_preparation(self, task_id)

    def recover_unscored(self, task_id, reason):
        from .unscored_rehost import prepare_unscored_rehost
        return prepare_unscored_rehost(self,task_id,reason)

    def resume_unscored_preparation(self, task_id):
        from .unscored_rehost import resume_unscored_preparation
        return resume_unscored_preparation(self,task_id)

    def recover_evaluation(self, task_id, request_id, reason):
        from .evaluation_recovery import prepare_evaluation_recovery
        return prepare_evaluation_recovery(self, task_id, request_id, reason)

    def resume_evaluation_preparation(self, task_id):
        from .evaluation_recovery import resume_evaluation_preparation
        return resume_evaluation_preparation(self, task_id)

    def recover_final(self, task_id, reason):
        """Explicit new independent attempt after a proven infrastructure loss.

        This never retries a complete group or an invalid kernel. Earlier
        decisions in an infrastructure-interrupted group are retained, including
        losses; the new group is an explicitly separate measurement context.
        """
        self._require_active_study()
        if not reason or len(reason.strip()) < 20:
            raise ValueError("Describe the concrete infrastructure incident in at least twenty characters")
        with self.journal.controller():
            task = next((row for row in self.journal.tasks("final") if row["id"] == task_id), None)
            if not task or task["state"] != "recovery_required":
                raise ValueError("Only an incomplete final group can receive explicit infrastructure recovery")
            directory = self._directory(task)
            if (directory / "worker/slurm-diagnostics/quarantined.json").exists():
                raise ValueError("A CUDA/context quarantine is not eligible for infrastructure-only final recovery")
            evidence = [task.get("result") or {}]
            for name in ("failure.json", "group-result.pending.json"):
                if (directory / name).exists():
                    evidence.append(read_json(directory / name))
            evidence += [read_json(path) for path in (directory / "qualification").glob("*.json")]
            records = list(_walk(evidence))
            if any(row.get("status") in {"numerical_error", "memory_error", "contract_violation", "compilation_error",
                    "sanitizer_qualification_failed", "validated_no_gain"} for row in records):
                raise ValueError("A numerical, contract or sanitizer failure cannot be retried as infrastructure")
            prior_decisions = [{key: row.get(key) for key in ("evaluation_id", "status", "interval", "artifact_hash", "incumbent_artifact_hash")}
                for row in records if row.get("interval") and row.get("status") in {"promoted", "not_improved", "inconclusive"}]
            categories = {row.get("infrastructure_category") for row in records}
            if not categories & {"transport_timeout", "ssh_transport_failure", "worker_terminal_incomplete"}:
                raise ValueError("No qualified transport/worker infrastructure incident was recorded")
            if not task.get("job_id"):
                raise ValueError("Unknown scheduler ownership must be resolved before a new allocation")
            lock_dir = self.directory / "task-locks"
            lock_dir.mkdir(exist_ok=True)
            with (lock_dir / (hashlib.sha256(task_id.encode()).hexdigest()+".lock")).open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError("The previous final worker process is still active") from error
                config = config_from_dict(task["payload"][0]["config"])
                backend = self._backend(task, config, directory)
                backend.job_id = str(task["job_id"])
                terminal = backend.status()
                from .execution_backend import TERMINAL_STATES, slurm_state
                if slurm_state(terminal) not in TERMINAL_STATES or str(terminal.get("job_id")) != str(task["job_id"]):
                    raise ValueError("Previous owned GPU job is not confirmed terminal")
                # Reauthenticate every generator source before committing a new
                # private input distribution. No model call is made here.
                generations = {row["id"]: row for row in self.journal.tasks("generation")}
                for entry in task["payload"]:
                    generation = generations[entry["config"]["run_id"]]["result"] or {}
                    if generation.get("candidate"):
                        load_candidate(generation["candidate"])
                seed = secrets.randbits(63)
                commitment = hashlib.sha256(str(seed).encode()).hexdigest()
                session = "suite-final-" + secrets.token_hex(16)
                with self.journal.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    old = db.execute("SELECT seed,commitment FROM sealed_groups WHERE id=?",(task_id,)).fetchone()
                    snapshot = {"task":task,"evidence":evidence,"observed_metrics":observed_metrics(directory, task.get("result") or {}),
                                "prior_statistical_decisions":prior_decisions,"terminal":terminal}
                    db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)",(task_id, task["attempt"],
                        json.dumps(snapshot,sort_keys=True), old["seed"], old["commitment"], reason, time.time()))
                    db.execute("UPDATE sealed_groups SET seed=?,commitment=? WHERE id=?",(str(seed),commitment,task_id))
                    db.execute("UPDATE tasks SET state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=attempt+1 WHERE id=?",(session,task_id))
                audit = {"task_id":task_id,"previous_attempt":task["attempt"],"new_attempt":task["attempt"]+1,
                    "reason":reason,"infrastructure_categories":sorted(str(v) for v in categories if v),
                    "previous_session":task["session"],"previous_job_terminal":terminal,
                    "previous_seed_commitment":old["commitment"],"new_seed_commitment":commitment,
                    "new_session":session,"model_calls":0,"new_job_submitted":False,
                    "prior_statistical_decisions_retained":prior_decisions,
                    "raw_timings_pooled_between_attempts":False}
                write_json(self.directory/"recovery-audit"/(task_id+".%03d.json"%(task["attempt"]+1)),audit)
                write_json(self.directory/"final-seed-commitments.json",self.journal.commitments())
                return audit

    def status(self):
        tasks = self.journal.tasks()
        manifests = {r["id"]: r["result"] for r in tasks if r["kind"] == "generation" and r["result"]}
        groups = {r["id"]: r["result"] for r in tasks if r["kind"] == "final" and r["result"]}
        protocol_incidents, incident_observations = load_protocol_incidents(self.directory, self.plan)
        from .artifact_qualification_incidents import load_artifact_qualification_incidents
        artifact_incidents, artifact_observations = load_artifact_qualification_incidents(self.directory,
            [entry["config"]["run_id"] for entry in self.plan["entries"]])
        summary = summarize_suite(self.plan, manifests, groups, protocol_incidents=protocol_incidents,
                                  protocol_incident_observations=incident_observations,
                                  artifact_qualification_incidents=artifact_incidents,
                                  artifact_incident_observations=artifact_observations)
        with self.journal.connection() as db:
            previous_attempts = [{"task_id": row["task_id"], "attempt": row["attempt"], "reason": row["reason"],
                "seed_commitment": row["commitment"], "receipt": json.loads(row["snapshot"])}
                for row in db.execute("SELECT * FROM task_attempts ORDER BY task_id,attempt")]
        return {"schema_version": 1, "tasks": [{k: r[k] for k in ("id", "kind", "state", "session", "job_id", "error")} for r in tasks],
                "previous_attempts": previous_attempts,
                "final_groups": {key: {name: result.get(name) for name in (
                    "task_evidence_path", "environment", "contract_hash", "worker_terminal", "metrics",
                    "total_task_wall_seconds_including_queue", "execution_complete", "all_modes_available")}
                    | {"method_comparisons": {name: {field: row.get(field) for field in ("status", "ok", "interval", "artifact_hash", "incumbent_artifact_hash")}
                         for name, row in result.get("method_comparisons", {}).items()}} for key, result in groups.items()},
                "summary": summary, "concurrency": self.concurrency, "global_resource_caps": None}

    def run(self):
        self._require_active_study()
        with self.journal.controller():
            # Reuse only authenticated complete receipts. A changed source or
            # failed terminal state cannot be hidden by a cached journal flag.
            for task in self.journal.tasks():
                if task["state"] != "completed":
                    continue
                result = task["result"] or {}
                terminal = result.get("worker_terminal", {})
                if not (result.get("execution_complete") and terminal.get("terminal") and terminal.get("state") == "COMPLETED"
                        and str(terminal.get("exit_code")) == "0:0" and result.get("session_id") == task["session"]
                        and str(terminal.get("job_id")) == str(task["job_id"])):
                    raise ValueError("Completed task lacks a successful owned terminal receipt")
                if task["kind"] == "generation":
                    load_candidate(result["candidate"])
                else:
                    load_candidate(result["baseline"])
                    for value in result.get("candidates", {}).values():
                        if value.get("source_path"):
                            load_candidate(value)
            # The original optimizer temporarily installs its pinned direct
            # imports. Separate spawned Python processes keep those aliases,
            # LangGraph stores and model transports isolated between methods.
            pool = ThreadPoolExecutor if self.backend_factory else ProcessPoolExecutor
            options = {} if self.backend_factory else {"mp_context": multiprocessing.get_context("spawn")}
            from .dispatch_attestation import capture_dispatch_attestation
            self._controller_epoch = capture_dispatch_attestation(self.directory)

            def recorded_storage_failure(recorded):
                if not recorded or recorded["state"] != "recovery_required":
                    return False
                outcome = recorded.get("result") or {}
                return ((isinstance(outcome, dict) and outcome.get("infrastructure_category") == "remote_storage_failure")
                        or infrastructure_category(recorded.get("error", "")) == "remote_storage_failure")

            with pool(max_workers=self.concurrency, **options) as workers:
                # Stable sessions recover active work; known failures are retained.
                storage_failure = any(recorded_storage_failure(row) for row in self.journal.tasks())
                for kind in ("generation", "final"):
                    if storage_failure:
                        break
                    tasks = self.journal.tasks(kind)
                    if kind == "final":
                        generations = {r["id"]: r for r in self.journal.tasks("generation")}
                        tasks = [r for r in tasks if all(generations[e["config"]["run_id"]]["state"] in TERMINAL for e in r["payload"])]
                    initial = iter(row for row in sorted(tasks, key=self._dispatch_key)
                                   if row["state"] in {"pending", "running"})
                    futures = {}

                    def replenish():
                        while len(futures) < self.concurrency:
                            row = next(initial, None)
                            if row is None:
                                break
                            futures[workers.submit(self._task, row)] = row["id"]

                    replenish()
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        # Inspect all outcomes already available before filling
                        # any freed slot. Never cancel or signal active workers.
                        for future in done:
                            task_id = futures.pop(future)
                            future.result()
                            recorded = self.journal.task(task_id)
                            if recorded_storage_failure(recorded):
                                storage_failure = True
                            write_json(self.directory / "status.json", self.status())
                        if not storage_failure:
                            replenish()
        result = self.status()
        write_json(self.directory / "status.json", result)
        return result
