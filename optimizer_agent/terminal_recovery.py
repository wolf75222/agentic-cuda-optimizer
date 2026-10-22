"""Explicit terminal reconciliation of an already completed agent generation.

This changes lifecycle metadata only. It never retries generation, GPU work,
model calls, scheduler submission or worker shutdown. The original result and
failure bytes remain in an immutable preparation alongside its terminal proof.
An incomplete preparation resumes the same transaction, not a new attempt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time


class TerminalRecoveryRefused(ValueError):
    pass


LIFECYCLE_FIELDS = frozenset({"worker_terminal", "execution_complete",
    "task_started_at_unix", "task_finished_at_unix",
    "total_task_wall_seconds_including_queue", "queue_and_worker_start_wall_seconds"})
MAX_PUBLIC_BYTES = 8 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise TerminalRecoveryRefused(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _safe(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    _require(".." not in path.parts and path.is_relative_to(root), "Terminal evidence is outside the study")
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "Aliased terminal evidence")
    return path


def _raw(path, root):
    path = _safe(path, root)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and before.st_size <= MAX_PUBLIC_BYTES, "Terminal evidence is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_PUBLIC_BYTES + 1)
        after = os.fstat(descriptor)
        _require(len(raw) <= MAX_PUBLIC_BYTES and (before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_ino, after.st_size, after.st_mtime_ns), "Terminal evidence changed during reading")
        return raw
    finally:
        os.close(descriptor)


def _object(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            _require(key not in value, "Duplicate terminal evidence field")
            value[key] = child
        return value
    def invalid(value):
        raise TerminalRecoveryRefused("Nonfinite terminal evidence: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _require(isinstance(value, dict), "Terminal evidence must be a JSON object")
    return value


def _read(path, root):
    return _object(_raw(path, root))


def _publish_raw(path, raw, root):
    from .dispatch_attestation import _mkdir_durable, _write_immutable, _fsync_directory
    path = _safe(path, root)
    _mkdir_durable(path.parent)
    if path.exists():
        _require(_raw(path, root) == raw, "Immutable terminal backup differs")
        _fsync_directory(path.parent)
    else:
        _write_immutable(path, raw)


def _write(path, value, root):
    _publish_raw(path, json.dumps(value, indent=2, allow_nan=False).encode(), root)


def _install_accounting(directory, folder, prepared, root):
    """Publish the real cached scheduler observation after preserving old bytes."""
    from .dispatch_attestation import _mkdir_durable, _fsync_directory
    path = _safe(directory / "worker/slurm-diagnostics/accounting.json", root)
    record = prepared["accounting"]
    _require(isinstance(record, dict) and set(record) == {"path", "previous_sha256", "observed_sha256", "scope"},
             "Malformed accounting proof")
    _require(record["path"] == str(path.relative_to(root)), "Accounting path differs")
    observed = _raw(folder / "accounting.observed.json", root)
    _require(_sha(observed) == record["observed_sha256"]
             and _object(observed) == prepared["terminal"], "Observed accounting proof differs")
    previous = record["previous_sha256"]
    if previous is not None:
        _require(_sha(_raw(folder / "accounting.original.json", root)) == previous,
                 "Original accounting backup differs")
    existing = _sha(_raw(path, root)) if path.exists() else None
    _require(existing in {previous, record["observed_sha256"]}, "Accounting changed outside this transaction")
    if existing == record["observed_sha256"]:
        _fsync_directory(path.parent)  # Finish a preceding replace whose fsync failed.
        return
    _mkdir_durable(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".terminal-accounting-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(observed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _binding(task):
    return {"task_id": task["id"], "kind": task["kind"], "session_id": task["session"],
            "job_id": str(task.get("job_id")), "attempt": task.get("attempt", 0),
            "payload_sha256": _digest(task["payload"])}


def _science(result):
    return {key: value for key, value in result.items() if key not in LIFECYCLE_FIELDS}


def _terminal(proof, job):
    _require(isinstance(proof, dict) and proof.get("terminal") is True
             and proof.get("state") == "COMPLETED" and str(proof.get("exit_code")) == "0:0"
             and str(proof.get("job_id")) == str(job), "Owned worker must be terminal COMPLETED 0:0")


def _result(orchestrator, task, directory):
    from .suite_orchestrator import load_candidate
    result = _read(directory / "result.json", orchestrator.directory)
    _require(result.get("run_id") == task["id"] and result.get("session_id") == task["session"]
             and str(result.get("job_id")) == str(task.get("job_id")), "Completed result identity differs")
    _require(result.get("generation_complete") is True and result.get("generation_only") is True
             and result.get("method_status") == "completed" and result.get("status") == "generation_complete",
             "Only a completed generation-only agent method can be reconciled")
    _require(isinstance(result.get("candidate"), dict), "Completed result has no frozen candidate")
    candidate = result["candidate"]
    source = _safe(Path(candidate.get("source_path", "")), directory / "generation")
    _require(source == directory / "generation/best.cu", "Unexpected retained agent source role")
    source_raw = _raw(source, directory / "generation")
    _require(_sha(source_raw) == candidate.get("source_hash"), "Frozen candidate source changed")
    loaded = load_candidate(candidate)
    _require(result.get("best_candidate_id") == loaded.candidate_id
             and result.get("artifact_hash") == loaded.artifact_hash, "Retained candidate identity differs")
    contract = _read(directory / "generation/contract.json", orchestrator.directory)
    _require(isinstance(result.get("contract_hash"), str)
             and re.fullmatch(r"[0-9a-f]{64}", result["contract_hash"])
             and contract.get("contract_hash") == result["contract_hash"], "Generation contract differs")
    _require(not (directory / "generation/holdout-exposed.json").exists()
             and not (directory / "holdout-reveal.json").exists(), "Exposed final inputs cannot be reconciled here")
    existing = result.get("worker_terminal", {})
    _require(isinstance(existing, dict), "Malformed saved worker terminal metadata")
    if existing.get("terminal") is True:
        _terminal(existing, task["job_id"])
        _require(result.get("execution_complete") is True, "Saved terminal result is not execution-complete")
    return result, source


def _completed(result, terminal):
    _terminal(result.get("worker_terminal"), terminal["job_id"])
    _require(result["worker_terminal"] == terminal and result.get("execution_complete") is True,
             "Reconciliation changed the terminal proof or did not complete")
    for key in ("task_started_at_unix", "task_finished_at_unix", "total_task_wall_seconds_including_queue"):
        value = result.get(key)
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 "Invalid reconciled lifecycle timing")
    _require(result["task_finished_at_unix"] >= result["task_started_at_unix"]
             and result["total_task_wall_seconds_including_queue"]
             == result["task_finished_at_unix"] - result["task_started_at_unix"], "Lifecycle elapsed time differs")


def _validate_preparation(orchestrator, task, directory, folder, prepared):
    root = orchestrator.directory
    _require(set(prepared) == {"schema_version", "kind", "created_at_unix", "reason", "task_binding",
        "original_journal_state", "original_journal_error", "completed_cli_receipts", "accounting",
        "preserved_files", "scientific_fields_sha256", "terminal", "public_files", "scope"}
             and type(prepared.get("schema_version")) is int and prepared["schema_version"] == 1
             and prepared.get("kind") == "completed_generation_terminal_preparation"
             and prepared.get("task_binding") == _binding(task), "Terminal preparation identity differs")
    _require(isinstance(prepared["preserved_files"], dict), "Malformed terminal backup roles")
    for name, pin in prepared["preserved_files"].items():
        _require(isinstance(pin, dict) and set(pin) == {"backup", "sha256"}
                 and name in {"result.json", "failure.json"} and pin["backup"] == name + ".original",
                 "Unexpected terminal backup role")
        backup = _raw(folder / pin["backup"], root)
        _require(_sha(backup) == pin["sha256"], "Terminal backup changed")
        if name == "failure.json":
            _require(_raw(directory / name, root) == backup, "Historical failure changed")
    _require(set(prepared["preserved_files"]) == {"result.json", "failure.json"}, "Missing original result or failure")
    original = _object(_raw(folder / "result.json.original", root))
    _require(_digest(_science(original)) == prepared["scientific_fields_sha256"], "Original scientific projection differs")
    expected_paths = {str(p.relative_to(root)) for p in (directory / "generation/best.cu",
        directory / "generation/config.json", directory / "generation/contract.json",
        directory / "worker/slurm-session.json")}
    _require(isinstance(prepared["public_files"], list)
             and len(prepared["public_files"]) == len(expected_paths)
             and all(isinstance(pin, dict) for pin in prepared["public_files"])
             and {pin.get("path") for pin in prepared["public_files"]} == expected_paths,
             "Unexpected public terminal evidence roles")
    for pin in prepared["public_files"]:
        _require(set(pin) == {"path", "sha256"}, "Malformed terminal public pin")
        _require(_sha(_raw(root / pin["path"], root)) == pin["sha256"], "Pinned candidate/config/ACK changed")
    from .evaluation_recovery import _closed_model_receipts
    _require(_closed_model_receipts(directory / "generation") == prepared["completed_cli_receipts"],
             "Completed CLI receipts changed")
    _terminal(prepared["terminal"], task["job_id"])
    current, _ = _result(orchestrator, task, directory)
    _require(_digest(_science(current)) == prepared["scientific_fields_sha256"], "Scientific result changed during reconciliation")
    # An enriched result can only be one produced by the frozen terminal path.
    if _sha(_raw(directory / "result.json", root)) != prepared["preserved_files"]["result.json"]["sha256"]:
        _completed(current, prepared["terminal"])
    return current


def _commit(task, directory, folder, prepared, root):
    return {"schema_version": 1, "kind": "completed_generation_terminal_commit", "task_binding": _binding(task),
        "prepared_sha256": _sha(_raw(folder / "prepared.json", root)),
        "result_sha256": _sha(_raw(directory / "result.json", root)),
        "failure_sha256": _sha(_raw(directory / "failure.json", root)),
        "accounting_sha256": _sha(_raw(directory / "worker/slurm-diagnostics/accounting.json", root)),
        "scientific_fields_sha256": prepared["scientific_fields_sha256"], "worker_terminal": prepared["terminal"],
        "new_job_submitted": False, "model_calls_during_reconciliation": 0, "gpu_requests_during_reconciliation": 0,
        "scope": "Terminal lifecycle reconciliation only; no new scientific or performance qualification"}


def reconcile_completed_generation(orchestrator, task_id, reason):
    """Finish only terminal metadata, or resume this exact prepared transaction.

    Initial admission requires recovery_required and an already completed agent
    generation. Native/original CLI methods are outside this narrow API. A
    committed replay is read-only and performs no scheduler/network operation.
    """
    from .evaluation_recovery import _locked, _closed_model_receipts
    from .suite_orchestrator import config_from_dict, _local_session_job
    orchestrator._require_active_study()
    _require(isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id),
             "Invalid generation task identity")
    _require(isinstance(reason, str) and len(reason.strip()) >= 20, "Describe the terminal infrastructure incident")
    with orchestrator.journal.controller(), _locked(orchestrator, task_id):
        task = orchestrator.journal.task(task_id)
        _require(task is not None and task["kind"] == "generation" and isinstance(task["payload"], dict),
                 "Only generation tasks have terminal reconciliation")
        _require(task["payload"].get("mode") not in {"native_codex_vanilla", "true_upstream"},
                 "CLI methods require a separate terminal reconciliation protocol")
        _require(isinstance(task.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", task["job_id"]),
                 "No unambiguous owned scheduler job")
        _require(type(task.get("attempt", 0)) is int and task.get("attempt", 0) >= 0,
                 "Invalid generation attempt")
        directory = orchestrator._directory(task)
        folder = orchestrator.directory / "terminal-reconciliations" / _sha(task_id.encode()) / ("%03d" % task.get("attempt", 0))
        prepared_path, committed_path = folder / "prepared.json", folder / "committed.json"
        _safe(folder, orchestrator.directory)
        _safe(prepared_path, orchestrator.directory)
        _safe(committed_path, orchestrator.directory)
        prepared = _read(prepared_path, orchestrator.directory) if prepared_path.exists() else None
        if prepared:
            current = _validate_preparation(orchestrator, task, directory, folder, prepared)
            if committed_path.exists():
                commit = _read(committed_path, orchestrator.directory)
                _completed(current, prepared["terminal"])
                _require(_digest(commit) == _digest(_commit(task, directory, folder, prepared, orchestrator.directory))
                         and commit["accounting_sha256"] == prepared["accounting"]["observed_sha256"]
                         and task["state"] == "completed" and _digest(task["result"]) == _digest(current),
                         "Committed terminal reconciliation differs")
                return commit
            _require(task["state"] in {"running", "recovery_required", "completed"}, "Prepared task state changed")
        else:
            _require(not committed_path.exists() and task["state"] == "recovery_required", "No incomplete terminal recovery to reconcile")
            current, source = _result(orchestrator, task, directory)
            _require(isinstance(task.get("result"), dict)
                     and _digest(_science(task["result"])) == _digest(_science(current)),
                     "Journal does not preserve this completed scientific result")
            failure_raw = _raw(directory / "failure.json", orchestrator.directory)
            failure = _object(failure_raw)
            _require(failure.get("session_id") == task["session"] and str(failure.get("job_id")) == task["job_id"]
                     and failure.get("status") == "failed" and failure.get("execution_complete") is False,
                     "Historical failure ownership differs")
            _require(isinstance(failure.get("partial_result"), dict)
                     and _digest(_science(failure["partial_result"])) == _digest(_science(current)),
                     "Failure does not preserve this completed scientific result")
            config = config_from_dict(task["payload"]["config"])
            _require(config.run_id == task_id and _digest(_read(directory / "generation/config.json", orchestrator.directory))
                     == _digest(json.loads(json.dumps(config.to_dict()))), "Completed generation configuration differs")
            models = _closed_model_receipts(directory / "generation")
        # A crash after the result/journal update needs only its local commit.
        already_complete = prepared and task["state"] == "completed"
        if already_complete:
            _completed(current, prepared["terminal"])
            _require(_digest(task["result"]) == _digest(current), "Completed journal/result differs")
            _install_accounting(directory, folder, prepared, orchestrator.directory)
        else:
            config = config_from_dict(task["payload"]["config"])
            backend = orchestrator._backend(task, config, directory)
            local_job = _local_session_job(task, backend)
            _require(local_job == task["job_id"], "Local ACK differs from owned task")
            terminal = backend.attach_existing()  # Historical ownership/scheduler read only.
            _require(_local_session_job(task, backend) == local_job, "Attached worker ownership changed")
            _terminal(terminal, local_job)
            if not prepared:
                result_raw = _raw(directory / "result.json", orchestrator.directory)
                _require(_digest(_object(result_raw)) == _digest(current), "Completed result changed before preparation")
                accounting_path = _safe(directory / "worker/slurm-diagnostics/accounting.json", orchestrator.directory)
                previous_accounting = _raw(accounting_path, orchestrator.directory) if accounting_path.exists() else None
                preserved = {}
                for name, data in (("result.json", result_raw), ("failure.json", failure_raw)):
                    _publish_raw(folder / (name + ".original"), data, orchestrator.directory)
                    preserved[name] = {"backup": name + ".original", "sha256": _sha(data)}
                public_paths = [source, directory / "generation/config.json", directory / "generation/contract.json",
                                Path(backend.results_dir) / "slurm-session.json"]
                if previous_accounting is not None:
                    _publish_raw(folder / "accounting.original.json", previous_accounting, orchestrator.directory)
                observed_accounting = json.dumps(terminal, indent=2, allow_nan=False).encode()
                _publish_raw(folder / "accounting.observed.json", observed_accounting, orchestrator.directory)
                prepared = {"schema_version": 1, "kind": "completed_generation_terminal_preparation",
                    "created_at_unix": time.time(), "reason": reason, "task_binding": _binding(task),
                    "original_journal_state": task["state"], "original_journal_error": task.get("error"),
                    "completed_cli_receipts": models,
                    "accounting": {"path": str(accounting_path.relative_to(orchestrator.directory)),
                        "previous_sha256": _sha(previous_accounting) if previous_accounting is not None else None,
                        "observed_sha256": _sha(observed_accounting),
                        "scope": "Exact scheduler observation returned by authenticated attach_existing; not a new GPU request"},
                    "preserved_files": preserved, "scientific_fields_sha256": _digest(_science(current)),
                    "terminal": terminal, "public_files": [{"path": str(p.relative_to(orchestrator.directory)),
                    "sha256": _sha(_raw(p, orchestrator.directory))} for p in public_paths],
                    "scope": "Lifecycle metadata only; original observations/costs/candidate/method preserved; no execution or qualification"}
                _write(prepared_path, prepared, orchestrator.directory)
            else:
                _require(terminal == prepared["terminal"], "Terminal observation differs from prepared proof")
            _validate_preparation(orchestrator, task, directory, folder, prepared)
            _install_accounting(directory, folder, prepared, orchestrator.directory)
            # The shared lifecycle-only method has no generic execution catch:
            # ordinary I/O errors cannot overwrite the historical failure.
            orchestrator._commit_terminal_result(task, current, prepared["terminal"],
                task.get("started") or time.time(), None, owned_job_id=local_job)
            current_task = orchestrator.journal.task(task_id)
            current = _validate_preparation(orchestrator, current_task, directory, folder, prepared)
            _require(current_task["state"] == "completed" and _digest(current_task["result"]) == _digest(current),
                     "Terminal execution path did not durably complete the journal")
        _completed(current, prepared["terminal"])
        commit = _commit(task, directory, folder, prepared, orchestrator.directory)
        _write(committed_path, commit, orchestrator.directory)
        return commit
