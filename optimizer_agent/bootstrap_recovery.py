"""Explicit infrastructure recovery before the first generator/model call.

No submission, CUDA execution or model call occurs here. A live owned lease is
retained; a terminal lease receives an independent attempt, without pooling its
measurements. Incomplete/uncertain model calls are never treated as bootstrap.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import time


def prove_pre_model(directory, mode):
    """Fail closed on even an incomplete model intent or graph checkpoint."""
    directory = Path(directory)
    generation = directory / "generation"
    for name in ("model", "model-transport", "research", "trials"):
        path = generation / name
        if path.exists() and any(p.is_file() for p in path.rglob("*")):
            raise ValueError(f"Generator artifacts already exist in {name}; bootstrap recovery is forbidden")
    for name in ("langgraph.sqlite", "original-langgraph.sqlite", "experiments.sqlite", "manifest.json", "best.cu", "candidate.json"):
        if (generation / name).exists():
            raise ValueError(f"Generator/checkpoint artifact {name} requires its dedicated resume protocol")
    if mode == "native_codex_vanilla":
        receipt = generation / "evidence/receipt.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if (row.get("model_started") is True or row.get("child_pid") or row.get("thread_ids") or row.get("actual_models")
                    or row.get("usage_events") or any(row.get("usage", {}).values())
                    or row.get("status") not in {"starting", "failed"}):
                raise ValueError("An existing native prompt/process is never repeated by bootstrap recovery")
        for name in ("codex.jsonl", "last-message.json", "response.json", "candidate.json"):
            if (generation / "evidence" / name).exists():
                raise ValueError("Native model/output intent exists; preserve the one-prompt attempt")
        # Model tools cannot have produced child content before their Popen.
        workspace = generation / "workspace"
        if workspace.exists() and any(p.name not in {"baseline.cu", "GPU_TOOLS.md"} for p in workspace.iterdir()):
            raise ValueError("Unexpected native workspace artifacts require manual provenance review")
    return {"model_started": False, "generator_checkpoint_exists": False,
            "scope": "Only pre-model infrastructure bootstrap; not generic model replay"}


def _inventory(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and not path.is_symlink()}


def recover_pre_model(orchestrator, task_id, reason):
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import (config_from_dict, infrastructure_category,
                                     remote_storage_failure_directory, read_json, write_json)
    orchestrator._require_active_study()
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError("Describe the concrete infrastructure incident in at least twenty characters")
    with orchestrator.journal.controller():
        task = next((row for row in orchestrator.journal.tasks("generation") if row["id"] == task_id), None)
        if not task or task["state"] != "recovery_required":
            raise ValueError("Only a failed, inactive generation can receive bootstrap recovery")
        lock = orchestrator.directory / "task-locks" / (hashlib.sha256(task_id.encode()).hexdigest() + ".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("The previous generation process is still active") from error
            directory = orchestrator._directory(task)
            existing_intent = directory / "bootstrap-recovery.json"
            if existing_intent.exists():
                raise ValueError("A bootstrap recovery intent already exists; reconcile it instead of preparing twice")
            proof = prove_pre_model(directory, task["payload"]["mode"])
            for request_path in (directory / "request-receipts").glob("*.json"):
                request_receipt = read_json(request_path)
                if "report" in request_receipt and request_receipt["report"].get("ok") is not True:
                    raise ValueError("A failed native/precision/memory result cannot be retried as transport bootstrap")
            failure_path = directory / "failure.json"
            if not failure_path.is_file():
                raise ValueError("Bootstrap recovery requires the original failure receipt")
            failure = read_json(failure_path)
            category = failure.get("infrastructure_category") or infrastructure_category(failure.get("error", ""))
            if category not in {"transport_timeout", "ssh_transport_failure", "worker_terminal_incomplete", "remote_storage_failure"}:
                raise ValueError("The original failure is not a qualified transport/worker infrastructure incident")
            config = config_from_dict(task["payload"]["config"])
            backend = orchestrator._backend(task, config, directory)
            if (category == "remote_storage_failure"
                    and remote_storage_failure_directory(failure.get("error", "")) != backend.remote_dir):
                raise ValueError("Storage failure directory differs from the owned worker session")
            # Inspect remote ownership before deciding that a new allocation is
            # permitted. A lost sbatch ACK/intent is never interpreted as no job.
            remote = backend._json("job_id.json", optional=True)
            local_path = directory / "worker/slurm-session.json"
            if remote is None:
                markers = {name: backend._json(name, optional=True)
                           for name in ("submission_started.json", "submission_error.json")}
                if (task.get("job_id") or failure.get("job_id") or local_path.exists()
                        or (directory/"worker/hardware.json").exists()
                        or any((directory/"request-receipts").glob("*.json"))
                        or any(value is not None for value in markers.values())):
                    raise ValueError("Uncertain scheduler ownership must be reconciled before bootstrap recovery")
                status = {"state": "NOT_SUBMITTED", "terminal": False}
                live, terminal = False, False
            else:
                status = backend.attach_existing()
                if (task.get("job_id") is not None and str(task["job_id"]) != str(backend.job_id)):
                    raise ValueError("Remote owned job differs from the suite journal")
                if failure.get("job_id") is not None and str(failure["job_id"]) != str(backend.job_id):
                    raise ValueError("Remote owned job differs from the failure receipt")
                quarantined = backend._json("quarantined.json", optional=True)
                if quarantined is not None:
                    raise ValueError("CUDA quarantine is not a transport bootstrap incident")
                live, terminal = slurm_state(status) == "RUNNING", slurm_state(status) in TERMINAL_STATES
                if not live and not terminal:
                    raise ValueError("Owned worker is neither RUNNING nor confirmed terminal; preserve it")
            inventory = _inventory(directory)
            audit = {"schema_version": 1, "task_id": task_id, "reason": reason,
                "created_at_unix": time.time(), "previous_session": task["session"],
                "previous_job_id": backend.job_id, "ownership_status": status,
                "pre_model_proof": proof, "infrastructure_category": category,
                "previous_artifacts_sha256": inventory, "model_calls": 0,
                "new_job_submitted": False, "gpu_requests": 0,
                "same_live_allocation": live, "raw_timings_pooled_between_allocations": False,
                "phase": "prepared"}
            if live:
                from .bootstrap_replay import create_manifest
                hardware = read_json(directory / "worker/hardware.json")
                if str(hardware.get("slurm_job_id")) != str(backend.job_id):
                    raise ValueError("Bootstrap hardware belongs to a different owned allocation")
                ready = backend._json("ready.json")
                if not isinstance(ready, dict) or ready.get("ok") is not True or ready.get("hardware") != hardware:
                    raise ValueError("Live worker readiness/hardware differs from the frozen local receipt")
                manifest = create_manifest(directory / "request-receipts", hardware)
                manifest_path = directory / "bootstrap-replay-manifest.json"
                write_json(manifest_path, manifest)
                audit.update(replay_manifest=str(manifest_path), replay_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
                if task["payload"]["mode"] == "native_codex_vanilla" and (directory / "generation").exists():
                    audit["native_pre_model_archive"] = str(directory / "bootstrap-archives/000/generation")
                baseline = directory / "generation/baseline-measurement.json"
                if baseline.exists() and read_json(baseline).get("ok") is not True:
                    audit["failed_baseline_archive"] = str(directory / "bootstrap-archives/failed-baseline.json")
                # Commit the intent before moving any pre-model workspace.
                # An interrupted preparation is retained and cannot cause an
                # automatic second prompt or a second allocation.
                write_json(existing_intent, audit)
                # Native CLI insists on a fresh filesystem workspace. Its old
                # pre-model artifacts remain under the evaluator's denied root.
                if task["payload"]["mode"] == "native_codex_vanilla" and (directory / "generation").exists():
                    archive = directory / "bootstrap-archives" / "000"
                    archive.mkdir(parents=True, exist_ok=False)
                    (directory / "generation").rename(archive / "generation")
                    audit["native_pre_model_archive"] = str(archive / "generation")
                baseline = directory / "generation/baseline-measurement.json"
                if baseline.exists() and read_json(baseline).get("ok") is not True:
                    archive = directory / "bootstrap-archives" / "failed-baseline.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    baseline.rename(archive)
                write_json(existing_intent, audit)
                # "running" makes the existing graph/metadata resume path
                # explicit, with no new lease or generation identity.
                orchestrator.journal.update(task_id, state="running", job_id=str(backend.job_id), error=None, finished=None)
            elif terminal:
                session = "suite-gen-" + secrets.token_hex(16)
                audit.update(new_session=session, new_attempt=task["attempt"]+1,
                    previous_measurements_scope="Retained only; target identity changes require a fresh bootstrap")
                write_json(existing_intent, audit)
                with orchestrator.journal.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)", (task_id, task["attempt"],
                        json.dumps({"task":task,"bootstrap_recovery":audit},sort_keys=True), None, None, reason, time.time()))
                    db.execute("UPDATE tasks SET state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=attempt+1 WHERE id=?", (session,task_id))
            else:
                # Nothing was submitted and the generator never started. Stage
                # can finish under the exact stable session and worker intent.
                write_json(existing_intent, audit)
                orchestrator.journal.update(task_id, state="running", error=None, finished=None)
            audit["phase"] = "committed"
            write_json(existing_intent, audit)
            write_json(orchestrator.directory / "recovery-audit" / (task_id + ".bootstrap-%03d.json" % task["attempt"]), audit)
            return audit


def resume_preparation(orchestrator, task_id):
    """Finish only a durable preparation interrupted before its final ACK.

    This does not run the prepared generation. Remote ownership and every old
    artifact are checked again before the pending local transaction is finished.
    """
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import (config_from_dict, remote_storage_failure_directory,
                                     read_json, write_json)
    orchestrator._require_active_study()
    with orchestrator.journal.controller():
        task = next((r for r in orchestrator.journal.tasks("generation") if r["id"] == task_id), None)
        if not task or task["state"] not in {"pending", "running", "recovery_required"}:
            raise ValueError("No inactive bootstrap preparation can be resumed")
        lock = orchestrator.directory / "task-locks" / (hashlib.sha256(task_id.encode()).hexdigest()+".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("The prepared generation process is already active") from error
            directories = [orchestrator._directory(task)]
            if task["attempt"]:
                directories.append(orchestrator._directory({**task,"attempt":task["attempt"]-1}))
            intents = [(d,read_json(d/"bootstrap-recovery.json")) for d in directories
                       if (d/"bootstrap-recovery.json").exists()]
            prepared = [(d,a) for d,a in intents if a.get("phase") == "prepared" and a.get("task_id") == task_id]
            if len(prepared) != 1:
                raise ValueError("Exactly one uncommitted bootstrap preparation is required")
            directory,audit = prepared[0]
            old_task = {**task,"session":audit["previous_session"],"job_id":audit["previous_job_id"]}
            if task["session"] not in {audit["previous_session"],audit.get("new_session")}:
                raise ValueError("Prepared bootstrap and journal sessions differ")
            if (task["session"] == audit["previous_session"] and task.get("job_id") is not None
                    and str(task["job_id"]) != str(audit["previous_job_id"])):
                raise ValueError("Prepared bootstrap and journal job ownership differ")
            archive = Path(audit["native_pre_model_archive"]) if audit.get("native_pre_model_archive") else None
            baseline_archive = Path(audit["failed_baseline_archive"]) if audit.get("failed_baseline_archive") else None
            for moved in (archive,baseline_archive):
                if moved is not None and not moved.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("Prepared archive leaves its owned task directory")
            for name,sha in audit["previous_artifacts_sha256"].items():
                original = directory/name
                possible = [original]
                if archive is not None and name.startswith("generation/"):
                    possible.append(archive / name[len("generation/"):])
                if baseline_archive is not None and name == "generation/baseline-measurement.json":
                    possible.append(baseline_archive)
                actual = [p for p in possible if p.is_file() and not p.is_symlink()]
                if len(actual) != 1 or hashlib.sha256(actual[0].read_bytes()).hexdigest() != sha:
                    raise ValueError("Prepared bootstrap artifact changed, disappeared or was duplicated")
            # Re-run the pre-model guard against the retained directory, even
            # when its native workspace has already been atomically moved.
            prove_pre_model(directory,task["payload"]["mode"])
            if archive is not None and archive.exists():
                prove_pre_model(archive.parent,task["payload"]["mode"])
            config = config_from_dict(task["payload"]["config"])
            backend = orchestrator._backend(old_task,config,directory)
            if (audit.get("infrastructure_category") == "remote_storage_failure"
                    and remote_storage_failure_directory(read_json(directory / "failure.json").get("error", "")) != backend.remote_dir):
                raise ValueError("Prepared storage failure directory differs from the owned worker session")
            if audit["same_live_allocation"] or audit.get("new_session"):
                status = backend.attach_existing()
                if str(backend.job_id) != str(audit["previous_job_id"]):
                    raise ValueError("Prepared bootstrap remote ownership changed")
                if backend._json("quarantined.json",optional=True) is not None:
                    raise ValueError("Prepared CUDA context is now quarantined; preserve the incident")
                wanted = "RUNNING" if audit["same_live_allocation"] else None
                if (wanted and slurm_state(status) != wanted) or (not wanted and slurm_state(status) not in TERMINAL_STATES):
                    raise ValueError("Prepared worker state changed; preserve the intent for explicit reconciliation")
            else:
                if any(backend._json(name,optional=True) is not None for name in ("job_id.json","submission_started.json","submission_error.json")):
                    raise ValueError("An allocation appeared during pre-submission preparation")
            if audit["same_live_allocation"]:
                manifest = directory/"bootstrap-replay-manifest.json"
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != audit["replay_manifest_sha256"]:
                    raise ValueError("Prepared replay manifest changed")
                hardware = read_json(directory/"worker/hardware.json")
                ready = backend._json("ready.json")
                if ready.get("ok") is not True or ready.get("hardware") != hardware or str(hardware.get("slurm_job_id")) != str(backend.job_id):
                    raise ValueError("Prepared live hardware changed")
                for origin,destination in ((directory/"generation",archive),
                                           (directory/"generation/baseline-measurement.json",baseline_archive)):
                    if destination is not None and origin.exists():
                        if destination.exists():
                            raise ValueError("Both source and destination of prepared archive exist")
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        origin.rename(destination)
                orchestrator.journal.update(task_id,state="running",job_id=str(backend.job_id),error=None,finished=None)
            elif audit.get("new_session"):
                if task["session"] == audit["previous_session"]:
                    with orchestrator.journal.connection() as db:
                        db.execute("BEGIN IMMEDIATE")
                        db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)",(task_id,task["attempt"],
                            json.dumps({"task":task,"bootstrap_recovery":audit},sort_keys=True),None,None,audit["reason"],time.time()))
                        db.execute("UPDATE tasks SET state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=attempt+1 WHERE id=?",(audit["new_session"],task_id))
                elif task["attempt"] != audit["new_attempt"] or task["state"] != "pending":
                    raise ValueError("Prepared independent attempt has already advanced")
            else:
                orchestrator.journal.update(task_id,state="running",error=None,finished=None)
            audit.update(phase="committed",preparation_resumed_at_unix=time.time())
            write_json(directory/"bootstrap-recovery.json",audit)
            old_attempt = audit.get("new_attempt",task["attempt"]+1)-1
            write_json(orchestrator.directory/"recovery-audit"/(task_id+".bootstrap-%03d.json"%old_attempt),audit)
            return audit
