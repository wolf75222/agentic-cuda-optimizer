"""Explicit §11.2 recovery of uncertain canonical evaluations.

Preparation authenticates the old worker, holds its task lock, and copies only
the target cold campaign. It never launches a worker/model or decodes stored
RNG/checkpoint values. Old attempts, reservations and partial results remain
unchanged. A new allocation must still pass the runtime's exact environment,
contract and UUID checks before it can evaluate the same proposal.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import time
import uuid


class EvaluationRecoveryRefused(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _safe(path, root):
    path, root = Path(path), Path(root)
    if not path.is_relative_to(root):
        raise EvaluationRecoveryRefused("Recovery evidence is outside its owned directory")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise EvaluationRecoveryRefused("Symlinked recovery evidence is forbidden")
    if path.is_file() and path.stat().st_nlink != 1:
        raise EvaluationRecoveryRefused("Aliased recovery evidence is forbidden")
    return path


def _read(path, root):
    value = json.loads(_safe(path, root).read_bytes())
    if not isinstance(value, dict):
        raise EvaluationRecoveryRefused("Recovery evidence must be a JSON object")
    return value


def _sha(path, root):
    return hashlib.sha256(_safe(path, root).read_bytes()).hexdigest()


def _write(path, value):
    # The canonical writer also synchronizes newly created parent directories.
    from .campaign import write_immutable_json
    write_immutable_json(path, value)


def _inventory(root):
    root = Path(root)
    _safe(root, root)
    return {str(path.relative_to(root)): _sha(path, root)
            for path in sorted(root.rglob("*")) if _safe(path, root).is_file()}


def _sync_copy(root):
    for path in sorted(root.rglob("*")):
        if _safe(path, root).is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in [*sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True), root, root.parent]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _closed_model_receipts(generation):
    """Refuse an ambiguous child rather than infer liveness from a reused PID.

    A terminal returncode is the trusted CLI wrapper's observation of child
    completion. Prepared/started receipts without that observation, including
    a Popen-before-started-receipt crash, require separate investigation. No
    process command line, prompt, PID-only test or provider claim is used.
    """
    directory = generation / "model"
    records = []
    if not directory.exists():
        return records
    for path in sorted(directory.iterdir()):
        _safe(path, generation)
        if path.name.endswith(".tmp"):
            raise EvaluationRecoveryRefused("An unfinished CLI receipt publication is ambiguous")
        if re.fullmatch(r"[0-9a-f]{32}\.jsonl", path.name):
            if not path.with_suffix(".json").is_file():
                raise EvaluationRecoveryRefused("CLI output has no terminal invocation receipt")
        if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name):
            continue
        row = _read(path, generation)
        launch = row.get("cli_launch", {})
        if not isinstance(launch, dict) or row.get("call_id") != path.stem:
            raise EvaluationRecoveryRefused("CLI receipt identity is malformed")
        terminal = row.get("status") in {"completed", "failed", "timeout", "tool_budget_exhausted"}
        ended = type(row.get("returncode")) is int
        never_started = (launch.get("state") == "launch_failed" and not launch.get("child_pid"))
        if not terminal or not (ended or never_started):
            raise EvaluationRecoveryRefused("A target CLI may still be active; no authenticated child completion")
        records.append({"call_id": row["call_id"], "path": str(path.relative_to(generation)),
                        "sha256": _sha(path, generation), "status": row["status"],
                        "returncode": row.get("returncode"), "launch_failed_before_child": never_started})
    return records


def _request_record(generation, request_id):
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{64}", request_id):
        raise EvaluationRecoveryRefused("An exact canonical request SHA is required")
    authorization = generation / "evaluation-recoveries" / (request_id + ".json")
    if authorization.exists() or authorization.is_symlink():
        raise EvaluationRecoveryRefused("This historical request was already authorized for recovery; select the current uncertain tip")
    paths = list((generation / "trials").glob("*/evaluation-requests/" + request_id))
    if len(paths) != 1:
        raise EvaluationRecoveryRefused("The uncertain request is missing or ambiguous")
    directory = _safe(paths[0], generation)
    if (directory / "completion.json").exists():
        raise EvaluationRecoveryRefused("A completed request must use normal result replay, not recovery")
    request, intent = (_read(directory / name, generation) for name in ("request.json", "intent.json"))
    if (request.get("kind") != "campaign_evaluation_request" or request.get("schema_version") != 1
            or _digest(request) != request_id or request.get("trial_id") != directory.parent.parent.name
            or not re.fullmatch(r"[0-9a-f]{32}", str(intent.get("attempt_id", "")))):
        raise EvaluationRecoveryRefused("Uncertain canonical request/attempt identity mismatch")
    expected = {"request_id": request_id, "trial_id": request["trial_id"], "repair": request["repair"],
        "research_round": request["research_round"], "proposal_sha256": request["proposal"]["sha256"],
        "template_candidate_id": request["template"]["candidate_id"],
        "template_artifact_hash": request["template"]["artifact_hash"],
        "reservation_id": "evaluation-" + request_id}
    if any(intent.get(key) != value for key, value in expected.items()):
        raise EvaluationRecoveryRefused("Uncertain intent binding mismatch")
    proposal_name = request["proposal"].get("path")
    if (not isinstance(proposal_name, str) or not re.fullmatch(r"proposal-[0-9]+-[0-9]+\.json", proposal_name)
            or _sha(directory.parent.parent / proposal_name, generation) != request["proposal"]["sha256"]):
        raise EvaluationRecoveryRefused("Uncertain proposal provenance mismatch")
    return directory, request, intent


def _intent_path(orchestrator, task_id, attempt):
    name = hashlib.sha256(task_id.encode()).hexdigest() + "-%03d.json" % attempt
    return orchestrator.directory / "evaluation-recovery-intents" / name


def _worker_proof(task, status, quarantine):
    from .execution_backend import TERMINAL_STATES, slurm_state
    job = str(task.get("job_id"))
    if str(status.get("job_id")) != job:
        raise EvaluationRecoveryRefused("Previous scheduler receipt job mismatch")
    if slurm_state(status) in TERMINAL_STATES:
        return "authenticated_scheduler_terminal"
    reservation = quarantine.get("worker_reservation", {}) if isinstance(quarantine, dict) else {}
    if (isinstance(quarantine, dict) and quarantine.get("worker_poisoned") is True
            and str(quarantine.get("slurm_job_id")) == job and isinstance(reservation, dict)
            and reservation.get("session_id") == task["session"] and str(reservation.get("job_id")) == job):
        return "authenticated_remote_worker_quarantine"
    raise EvaluationRecoveryRefused("Previous owned worker is neither stopped nor authentically quarantined")


def _without_epoch(request):
    return {key: value for key, value in request.items() if key != "recovery"}


def _relocated_request(request, current_generation, previous_generation):
    """Compare file identities across a proved cold copy without hiding paths.

    Only manifest file keys below the copied generation may move. Options,
    external dependencies, hashes, architecture, ABI and toolchain are exact.
    The returned comparison value is never the new execution request.
    """
    from .compile_cache import content_hash
    value = json.loads(json.dumps(_without_epoch(request)))
    manifests = {}
    for manifest in value["compilation_inputs"].values():
        files = {}
        for name, sha in manifest["files"].items():
            path = Path(name)
            key = str(Path(previous_generation) / path.relative_to(current_generation)) if path.is_relative_to(current_generation) else name
            if key in files:
                raise EvaluationRecoveryRefused("Ambiguous compilation dependency relocation")
            files[key] = sha
        manifest["files"] = files
        manifest.pop("key")
        key = content_hash(manifest)
        manifests[key] = {**manifest, "key": key}
    value["compilation_inputs"] = manifests
    return value


def _epoch_binding(marker, marker_sha):
    return {"epoch_id": marker["epoch_id"], "previous_request_id": marker["previous_request_id"],
            "previous_attempt_id": marker["previous_attempt_id"], "authorization_sha256": marker_sha}


def recovery_request(generation, request):
    """Resolve a committed chain; preserve the NEW actual manifest paths."""
    generation = Path(generation)
    request = json.loads(json.dumps(request, default=str))
    records = {}
    for path in (generation / "evaluation-recoveries").glob("*.json"):
        if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            continue
        marker = _read(path, generation)
        previous = marker.get("request")
        if not isinstance(previous, dict):
            raise EvaluationRecoveryRefused("Malformed evaluation recovery request")
        if not all(previous.get(key) == request.get(key) for key in ("trial_id", "repair", "research_round")):
            continue  # A later scientific trial/repair remains an ordinary request.
        epoch, request_id = marker.get("epoch_id"), path.stem
        if (marker.get("schema_version") != 1 or marker.get("kind") != "uncertain_evaluation_recovery"
                or not isinstance(epoch, str) or not re.fullmatch(r"[0-9a-f]{32}", epoch)
                or marker.get("previous_request_id") != request_id or _digest(previous) != request_id
                or not isinstance(marker.get("previous_generation"), str)
                or not Path(marker["previous_generation"]).is_absolute()):
            raise EvaluationRecoveryRefused("Evaluation recovery request/epoch mismatch")
        directory = generation / "trials" / request["trial_id"] / "evaluation-requests" / request_id
        if _read(directory / "request.json", generation) != previous:
            raise EvaluationRecoveryRefused("Recovered request bytes changed")
        files = marker.get("old_request_files")
        if not isinstance(files, dict) or not {"request.json", "intent.json"} <= files.keys():
            raise EvaluationRecoveryRefused("Incomplete uncertain evaluation evidence")
        for name, expected in files.items():
            if name not in {"request.json", "intent.json", "constraints.json", "autotuning.json", "measurement.json"}:
                raise EvaluationRecoveryRefused("Unexpected uncertain receipt link")
            if _sha(directory / name, generation) != expected:
                raise EvaluationRecoveryRefused("Uncertain evaluation evidence changed")
        intent = _read(directory / "intent.json", generation)
        commit = _read(generation / "evaluation-recoveries" / ("committed-" + epoch + ".json"), generation)
        if (intent.get("attempt_id") != marker.get("previous_attempt_id")
                or set(commit) != {"epoch_id", "marker_sha256", "preparation_sha256", "new_session", "new_task_attempt"}
                or not re.fullmatch(r"[0-9a-f]{64}", str(commit.get("preparation_sha256", "")))
                or commit.get("epoch_id") != epoch or commit.get("marker_sha256") != _sha(path, generation)
                or commit.get("new_session") != marker.get("new_session")
                or commit.get("new_task_attempt") != marker.get("new_task_attempt")):
            raise EvaluationRecoveryRefused("Evaluation recovery is not durably committed")
        records[request_id] = (marker, _sha(path, generation))
    if not records:
        return request
    parents = set()
    for marker, _ in records.values():
        parent = marker["request"].get("recovery")
        if parent is None:
            continue
        if not isinstance(parent, dict) or parent.get("previous_request_id") not in records:
            raise EvaluationRecoveryRefused("Broken evaluation recovery chain")
        previous, sha = records[parent["previous_request_id"]]
        if (parent != _epoch_binding(previous, sha)
                or _relocated_request(marker["request"], Path(marker["previous_generation"]),
                                      Path(previous["previous_generation"])) != _without_epoch(previous["request"])):
            raise EvaluationRecoveryRefused("Evaluation recovery chain proposal mismatch")
        parents.add(parent["previous_request_id"])
    tips = set(records) - parents
    if len(tips) != 1:
        raise EvaluationRecoveryRefused("Ambiguous or cyclic evaluation recovery chain")
    tip = next(iter(tips))
    visited, cursor = set(), tip
    while cursor not in visited:
        visited.add(cursor)
        parent = records[cursor][0]["request"].get("recovery")
        if parent is None:
            break
        cursor = parent["previous_request_id"]
    if visited != set(records):
        raise EvaluationRecoveryRefused("Disconnected or cyclic evaluation recovery chain")
    marker, sha = records[tip]
    if (_without_epoch(request) != _without_epoch(marker["request"])
            and _relocated_request(request, generation, Path(marker["previous_generation"])) != _without_epoch(marker["request"])):
        raise EvaluationRecoveryRefused("Pending evaluation differs from its authorized recovery proposal")
    # An old completion appearing later is historical, never selected here.
    return {**request, "recovery": _epoch_binding(marker, sha)}


def _new_payload(audit):
    return {**audit["original_task"]["payload"], "source_recovery": {
        "kind": "uncertain_evaluation_retry", "epoch_id": audit["epoch_id"],
        "marker_path": str(Path(audit["new_generation"]) / "evaluation-recoveries" / (audit["request_id"] + ".json")),
        "preparation_path": audit["preparation_path"], "preparation_sha256": audit["preparation_sha256"]}}


def _ready_record(audit, preparation_sha, marker_sha):
    return {"epoch_id": audit["epoch_id"], "preparation_sha256": preparation_sha,
        "marker_sha256": marker_sha, "rng_or_checkpoint_values_decoded": False,
        "cold_copy_files_sha256": {**audit["source_files_sha256"],
            "evaluation-recoveries/" + audit["request_id"] + ".json": marker_sha}}


def validate_task_recovery(orchestrator, task):
    """Validate task-to-preparation binding before scheduler submission."""
    if task.get("kind") != "generation":
        return
    payload = task.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("source_recovery", {}), dict):
        raise EvaluationRecoveryRefused("Malformed generation recovery payload")
    binding = payload.get("source_recovery", {})
    if binding.get("kind") != "uncertain_evaluation_retry":
        return
    intent_path = _intent_path(orchestrator, task["id"], task["attempt"])
    if str(intent_path) != binding.get("preparation_path") or _sha(intent_path, orchestrator.directory) != binding.get("preparation_sha256"):
        raise EvaluationRecoveryRefused("Recovery task preparation pin mismatch")
    audit = _read(intent_path, orchestrator.directory)
    if (task["session"] != audit["new_session"] or task["attempt"] != audit["new_task_attempt"]
            or task["payload"] != _new_payload({**audit, "preparation_sha256": binding["preparation_sha256"]})
            or str(orchestrator._directory(task) / "generation") != audit["new_generation"]):
        raise EvaluationRecoveryRefused("Recovery task/session/configuration mismatch")
    generation = Path(audit["new_generation"])
    marker = generation / "evaluation-recoveries" / (audit["request_id"] + ".json")
    if str(marker) != binding.get("marker_path") or _read(marker, generation) != audit["marker"]:
        raise EvaluationRecoveryRefused("Recovery authorization marker mismatch")
    commit = _read(generation / "evaluation-recoveries" / ("committed-" + audit["epoch_id"] + ".json"), generation)
    if (commit != _read(intent_path.with_suffix(".committed.json"), orchestrator.directory)
            or commit.get("preparation_sha256") != binding["preparation_sha256"]):
        raise EvaluationRecoveryRefused("Recovery commit preparation pin mismatch")
    if _read(intent_path.with_suffix(".ready.json"), orchestrator.directory) != _ready_record(
            audit, binding["preparation_sha256"], _sha(marker, generation)):
        raise EvaluationRecoveryRefused("Recovery readiness evidence changed")
    recovery_request(generation, audit["marker"]["request"])


def preparation_pending(orchestrator, task):
    if task.get("kind") != "generation":
        return False
    for attempt in (task.get("attempt", 0), task.get("attempt", 0) + 1):
        path = _intent_path(orchestrator, task["id"], attempt)
        if path.exists():
            audit = _read(path, orchestrator.directory)
            if task["session"] in {audit["original_task"]["session"], audit["new_session"]}:
                if not path.with_suffix(".committed.json").exists():
                    return True
    return False


def _finish(orchestrator, audit, path):
    original = audit["original_task"]
    preparation_sha = _sha(path, orchestrator.directory)
    enriched = {**audit, "preparation_sha256": preparation_sha}
    expected_payload = _new_payload(enriched)
    current = orchestrator.journal.task(original["id"])
    transferred = (current and current["attempt"] == audit["new_task_attempt"]
                   and current["session"] == audit["new_session"] and current["payload"] == expected_payload)
    if not transferred:
        if current != original:
            raise EvaluationRecoveryRefused("The original recovery task advanced")
        source, destination = Path(audit["old_generation"]), Path(audit["new_generation"])
        if _inventory(source) != audit["source_files_sha256"]:
            raise EvaluationRecoveryRefused("The original cold campaign changed, including a possible late result")
        marker_path = destination / "evaluation-recoveries" / (audit["request_id"] + ".json")
        expected_copy = dict(audit["source_files_sha256"])
        if destination.exists():
            actual = _inventory(destination)
            if marker_path.exists() and _read(marker_path, destination) == audit["marker"]:
                actual.pop(str(marker_path.relative_to(destination)), None)
            if actual != expected_copy:
                archive = destination.parent / ("interrupted-evaluation-copy-" + uuid.uuid4().hex)
                destination.rename(archive)
                _write(archive.with_suffix(".preserved.json"), {"preparation_sha256": preparation_sha,
                    "path": str(archive), "scope": "Interrupted cold copy preserved; never a runnable recovery context"})
        if not destination.exists():
            from .campaign import _mkdir_evaluation_durable
            _mkdir_evaluation_durable(destination.parent)
            shutil.copytree(source, destination, copy_function=shutil.copy2)
        if _inventory(source) != expected_copy or {k: v for k, v in _inventory(destination).items()
                if k != str(marker_path.relative_to(destination))} != expected_copy:
            raise EvaluationRecoveryRefused("The cold copy differs from its authenticated source")
        _sync_copy(destination)
        _write(marker_path, audit["marker"])
        _write(path.with_suffix(".ready.json"), _ready_record(audit, preparation_sha, _sha(marker_path, destination)))
        with orchestrator.journal.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE id=?", (original["id"],)).fetchone()
            snapshot = {**dict(row), "payload": json.loads(row["payload"]),
                        "result": json.loads(row["result"]) if row["result"] else None} if row else None
            if snapshot != original:
                raise EvaluationRecoveryRefused("The owned task changed before recovery commit")
            db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)", (original["id"], original["attempt"],
                json.dumps({"task": original, "evaluation_recovery_preparation_sha256": preparation_sha}, sort_keys=True),
                None, None, audit["reason"], time.time()))
            db.execute("UPDATE tasks SET payload=?,state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=? WHERE id=?",
                (json.dumps(expected_payload, sort_keys=True), audit["new_session"], audit["new_task_attempt"], original["id"]))
    else:
        with orchestrator.journal.connection() as db:
            row = db.execute("SELECT snapshot FROM task_attempts WHERE task_id=? AND attempt=?",
                             (original["id"], original["attempt"])).fetchone()
        archived = json.loads(row[0]) if row else {}
        if archived != {"task": original, "evaluation_recovery_preparation_sha256": preparation_sha}:
            raise EvaluationRecoveryRefused("The original task snapshot does not authenticate this recovery")
    generation = Path(audit["new_generation"])
    marker = generation / "evaluation-recoveries" / (audit["request_id"] + ".json")
    ready = _read(path.with_suffix(".ready.json"), orchestrator.directory)
    if (ready != _ready_record(audit, preparation_sha, _sha(marker, generation))
            or _read(marker, generation) != audit["marker"]):
        raise EvaluationRecoveryRefused("Recovery readiness evidence changed")
    if not path.with_suffix(".committed.json").exists():
        actual = _inventory(generation)
        actual.pop("evaluation-recoveries/committed-" + audit["epoch_id"] + ".json", None)
        if actual != ready["cold_copy_files_sha256"]:
            raise EvaluationRecoveryRefused("The prepared cold context changed before activation")
    commit = {"epoch_id": audit["epoch_id"], "marker_sha256": _sha(marker, generation),
        "preparation_sha256": preparation_sha, "new_session": audit["new_session"],
        "new_task_attempt": audit["new_task_attempt"]}
    _write(generation / "evaluation-recoveries" / ("committed-" + audit["epoch_id"] + ".json"), commit)
    _write(path.with_suffix(".committed.json"), commit)
    return {**commit, "task_id": original["id"], "previous_request_id": audit["request_id"],
        "previous_attempt_id": audit["marker"]["previous_attempt_id"], "previous_cost_status": "uncertain_preserved",
        "new_job_submitted": False, "model_calls_during_preparation": 0, "gpu_requests_during_preparation": 0}


def _locked(orchestrator, task_id):
    directory = orchestrator.directory / "task-locks"
    directory.mkdir(exist_ok=True)
    handle = (directory / (hashlib.sha256(task_id.encode()).hexdigest() + ".lock")).open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise EvaluationRecoveryRefused("The target generation process is still active") from error
    return handle


def prepare_evaluation_recovery(orchestrator, task_id, request_id, reason):
    orchestrator._require_active_study()
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise EvaluationRecoveryRefused("Describe the interrupted evaluation in at least twenty characters")
    with orchestrator.journal.controller(), _locked(orchestrator, task_id):
        task = orchestrator.journal.task(task_id)
        if (not task or task["kind"] != "generation" or task["state"] not in {"running", "recovery_required"}
                or task["payload"]["mode"] in {"true_upstream", "native_codex_vanilla"}):
            raise EvaluationRecoveryRefused("Only an incomplete improved-agent generation evaluation can be recovered")
        path = _intent_path(orchestrator, task_id, task["attempt"] + 1)
        if path.exists():
            raise EvaluationRecoveryRefused("Preparation already exists; resume it explicitly")
        from .suite_orchestrator import config_from_dict, _local_session_job
        config = config_from_dict(task["payload"]["config"])
        directory = orchestrator._directory(task)
        generation = directory / "generation"
        if (generation / "manifest.json").exists() or not (generation / "langgraph.sqlite").is_file():
            raise EvaluationRecoveryRefused("A cold unfinished campaign checkpoint is required")
        if (_read(generation / "execution-mode.json", generation) != {"generation_only": True}
                or (generation / "holdout-exposed.json").exists()):
            raise EvaluationRecoveryRefused("Only an unexposed generation-only campaign can be copied")
        if _read(generation / "config.json", generation) != json.loads(json.dumps(config.to_dict())):
            raise EvaluationRecoveryRefused("Campaign configuration changed")
        request_directory, request, intent = _request_record(generation, request_id)
        if (request["run_id"] != task_id
                or _read(generation / "contract.json", generation).get("contract_hash") != request["contract_hash"]
                or _digest(_read(generation / "environment.json", generation)) != request["environment_fingerprint"]):
            raise EvaluationRecoveryRefused("Campaign contract/environment binding mismatch")
        models = _closed_model_receipts(generation)
        backend = orchestrator._backend(task, config, directory)
        terminal = backend.attach_existing()  # Read-only ownership + scheduler proof, never start/close.
        owned = _local_session_job(task, backend)
        if (not owned or owned != str(task.get("job_id")) or str(terminal.get("job_id")) != owned):
            raise EvaluationRecoveryRefused("Previous worker ownership is not authenticated")
        quarantine = backend._json("quarantined.json", optional=True)
        worker_proof = _worker_proof(task, terminal, quarantine)
        epoch = uuid.uuid4().hex
        next_task = {**task, "attempt": task["attempt"] + 1}
        destination = orchestrator._directory(next_task) / "generation"
        if destination.exists():
            raise EvaluationRecoveryRefused("The new task attempt directory is occupied")
        files = _inventory(generation)
        marker = {"schema_version": 1, "kind": "uncertain_evaluation_recovery", "epoch_id": epoch,
            "previous_request_id": request_id, "previous_attempt_id": intent["attempt_id"], "request": request,
            "previous_generation": str(generation),
            "old_request_files": {p.name: _sha(p, generation) for p in request_directory.iterdir()
                if p.is_file() and p.name in {"request.json", "intent.json", "constraints.json", "autotuning.json", "measurement.json"}},
            "previous_task_attempt": task["attempt"], "new_task_attempt": next_task["attempt"],
            "previous_session": task["session"], "previous_job_id": owned,
            "new_session": "suite-gen-" + secrets.token_hex(16),
            "previous_worker_observation": terminal, "previous_quarantine_observation": quarantine,
            "worker_proof_basis": worker_proof,
            "completed_cli_receipts": models, "previous_cost_status": "uncertain_preserved",
            "scope": "Explicit new attempt for the same proposal only; old result/attempt/reservation remain uncertain and unchanged. No score or cost is fabricated."}
        audit = {"schema_version": 1, "kind": "uncertain_evaluation_recovery_preparation", "epoch_id": epoch,
            "request_id": request_id, "reason": reason, "created_at_unix": time.time(), "original_task": task,
            "new_task_attempt": next_task["attempt"], "new_session": marker["new_session"],
            "old_generation": str(generation), "new_generation": str(destination),
            "source_files_sha256": files, "marker": marker, "preparation_path": str(path),
            "rng_or_checkpoint_values_decoded": False}
        _write(path, audit)
        return _finish(orchestrator, audit, path)


def resume_evaluation_preparation(orchestrator, task_id):
    orchestrator._require_active_study()
    with orchestrator.journal.controller(), _locked(orchestrator, task_id):
        task = orchestrator.journal.task(task_id)
        if not task:
            raise EvaluationRecoveryRefused("Unknown recovery task")
        paths = [_intent_path(orchestrator, task_id, n) for n in (task["attempt"], task["attempt"] + 1)]
        matching = [path for path in paths if path.exists() and not path.with_suffix(".committed.json").exists()]
        if not matching and task["payload"].get("source_recovery", {}).get("kind") == "uncertain_evaluation_retry":
            committed = _intent_path(orchestrator, task_id, task["attempt"])
            if committed.exists() and committed.with_suffix(".committed.json").exists():
                matching = [committed]
        if len(matching) != 1:
            raise EvaluationRecoveryRefused("No unique unfinished preparation; committed work is never recreated")
        path = matching[0]
        audit = _read(path, orchestrator.directory)
        # Recheck child completion and cold worker ownership before unfinished
        # copying. A journal-committed preparation only publishes final receipts.
        if task == audit["original_task"]:
            _closed_model_receipts(Path(audit["old_generation"]))
            from .suite_orchestrator import config_from_dict, _local_session_job
            backend = orchestrator._backend(task, config_from_dict(task["payload"]["config"]), orchestrator._directory(task))
            terminal = backend.attach_existing()
            quarantine = backend._json("quarantined.json", optional=True)
            if (_local_session_job(task, backend) != audit["marker"]["previous_job_id"]
                    or str(terminal.get("job_id")) != audit["marker"]["previous_job_id"]):
                raise EvaluationRecoveryRefused("Previous worker no longer authenticates a cold recovery")
            _worker_proof(task, terminal, quarantine)
        return _finish(orchestrator, audit, path)
