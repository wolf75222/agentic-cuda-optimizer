"""Explicit independent confirmation after a completed canonical evaluation.

No historical timing pair is replayed into the new epoch. Preparation reads a
cold checkpoint internally but emits only fixed public identifiers, never RNG
or checkpoint payloads. Submission and placement remain orchestrator duties.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
from types import SimpleNamespace
import time
import uuid

from . import evaluation_recovery as cold
from .experiment_store import ExperimentStore

NAMESPACE = "confirmation-recoveries"
KIND = "completed_confirmation_recovery"
PREPARATION = KIND + "_preparation"
SNAPSHOT_KEY = "confirmation_recovery_preparation_sha256"


class ConfirmationRecoveryRefused(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise ConfirmationRecoveryRefused(reason)


@contextmanager
def _readonly(path):
    path = Path(path)
    cold._safe(path, path.parent)
    _require(path.is_file(), "Missing cold database")
    _require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
             "Database has a live sidecar; cold recovery is required")
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        db.close()


class _ReadOnlyStore(ExperimentStore):
    """Use existing store identity APIs without schema creation or writes."""
    def __init__(self, path):
        self.path = Path(path)

    def connection(self):
        return _readonly(self.path)


def _no_execution(*args, **kwargs):
    raise ConfirmationRecoveryRefused("Checkpoint inspection cannot execute a graph node")


def _checkpoint(generation, run_id):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from .campaign_workflow import build_campaign_graph
    runtime = SimpleNamespace(max_repairs=0, **{name: _no_execution for name in
        ("initialize", "select", "research", "generate", "evaluate", "integrate", "diagnose", "finalize")})
    with _readonly(generation / "langgraph.sqlite") as db:
        saver = SqliteSaver(db)
        # Existing, cold schema only. setup() otherwise runs DDL even for reads.
        saver.is_setup = True
        snapshot = build_campaign_graph(runtime, saver).get_state({"configurable": {"thread_id": run_id}})
    return snapshot


def _record(generation, request_id):
    from .campaign import evaluation_record
    _require(isinstance(request_id, str) and re.fullmatch(r"[0-9a-f]{64}", request_id), "Exact canonical request SHA required")
    paths = list((generation / "trials").glob("*/evaluation-requests/" + request_id))
    _require(len(paths) == 1, "Completed canonical request missing or ambiguous")
    directory = cold._safe(paths[0], generation)
    request, completion, measurement = evaluation_record(directory)
    _require(measurement.get("ok") is True and measurement.get("status") == "measured", "Successful completed canonical measurement required")
    return directory, request, completion, measurement


def _candidate(store, generation, run_id, candidate_id):
    from .evaluation import Candidate, LaunchRule
    state = store.get_candidate_source(candidate_id, run_id)
    source = generation / "candidates" / (candidate_id + ".cu")
    _require(cold._safe(source, generation).read_text() == state["source"], "Archived candidate source changed")
    candidate = Candidate(candidate_id, source, state["kernel"], LaunchRule(**state["launch_rule"]), state["parameters"])
    _require(candidate.artifact_hash == state["artifact_hash"] and not state.get("current_validity", {}).get("valid") is False,
             "Archived candidate artifact or validity changed")
    return {"candidate_id": candidate_id, "artifact_hash": candidate.artifact_hash,
            "source_sha256": candidate._source_hash, "launch_rule": json.loads(json.dumps(asdict(candidate.launch_rule))), "parameters": candidate.parameters}


def checkpoint_projection(generation, request_id):
    """Trusted, read-only projection; no checkpoint values escape this helper."""
    generation = Path(generation)
    _, request, _, _ = _record(generation, request_id)
    return _projection(generation, request_id, _checkpoint(generation, request["run_id"]),
                       _ReadOnlyStore(generation / "experiments.sqlite"))


def _projection(generation, request_id, snapshot, store):
    # The runtime supplies its actual live graph snapshot/store; preparation
    # constructs its own strictly cold, read-only adapters. No CLI proof input.
    from . import campaign
    directory, request, completion, measurement = _record(generation, request_id)
    _require(cold._read(generation / "execution-mode.json", generation) == {"generation_only": True}
             and not (generation / "holdout-exposed.json").exists(), "Only unexposed generation-only checkpoints are recoverable")
    _require(not (generation / "manifest.json").exists() and not (directory / "confirmation.json").exists()
             and not (directory.parent.parent / "result.json").exists(), "Completed confirmation/integration must use normal replay")
    state = snapshot.values
    _require(tuple(snapshot.next) == ("integrate",) and state.get("run_id") == request["run_id"], "Checkpoint must be waiting at integrate")
    _require(state.get("trial_id") == request["trial_id"] and state.get("candidate_id") == measurement["candidate_id"]
             and state.get("evaluation_id") == measurement["evaluation_id"] and state.get("evaluation_status") == "measured"
             and state.get("valid") is True and state.get("research_ready") is True
             and not state.get("needs_research") and not state.get("error"), "Checkpoint differs from its completed canonical measurement")
    _require(state.get("repair") == request["repair"] and state.get("research_round") == request["research_round"], "Checkpoint proposal round changed")
    incumbent = store.incumbent(request["run_id"])
    with store.connection() as db:
        trial = db.execute("SELECT status,payload,result,evaluation_id FROM trials WHERE run_id=? AND trial_id=?",
                           (request["run_id"], request["trial_id"])).fetchone()
        claimed = db.execute("SELECT 1 FROM reward_claims WHERE run_id=? AND evaluation_id=?",
                             (request["run_id"], measurement["evaluation_id"])).fetchone()
    _require(trial and trial["status"] == "pending" and trial["result"] is None and trial["evaluation_id"] is None
             and not claimed, "Trial already integrated or canonical reward already claimed")
    trial_payload = cold._read(directory.parent.parent / "trial.json", generation)
    _require(json.loads(trial["payload"])["action"] == trial_payload["action"], "Frozen trial action differs from store")
    config = cold._read(generation / "config.json", generation)
    _require(config.get("run_id") == request["run_id"] and type(state.get("trials_done")) is int
             and state["trials_done"] >= 0, "Checkpoint or configuration campaign identity changed")
    proposal = request["proposal"]
    proposal_path = directory.parent.parent / proposal["path"]
    _require(cold._sha(proposal_path, generation) == proposal["sha256"] and cold._read(proposal_path, generation) == proposal["payload"],
             "Completed proposal provenance changed")
    data = proposal["payload"]
    experimental = config.get("experimental_mode")
    campaign.Proposal.from_dict(data, parent_id=trial_payload["parent"]["candidate_id"], source_parent_id=data["source_parent_id"],
        contract_hash=request["contract_hash"], family=trial_payload["action"],
        experimental_mode=experimental if experimental in ("ablation_no_research", "ablation_no_pmpp", "ablation_pmpp_permanent_only") else None)
    documentary = {}
    if experimental != "ablation_no_research":
        dossier_path = directory.parent.parent / "dossier.json"
        dossier = campaign.ResearchDossier.from_dict(cold._read(dossier_path, generation))
        dossier.research_ready(request["contract_hash"], trial_payload["action"], request["environment_fingerprint"])
        dossier.validate_proposal(data)
        documentary["dossier_sha256"] = cold._sha(dossier_path, generation)
    _require(cold._read(generation / "contract.json", generation)["contract_hash"] == request["contract_hash"]
             and cold._digest(cold._read(generation / "environment.json", generation)) == request["environment_fingerprint"],
             "Canonical contract or environment changed")
    environment = cold._read(generation / "environment.json", generation)
    identity_keys = ("uuid", "architecture", "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
    _require(all(environment.get(key) not in (None, "") for key in identity_keys), "Complete physical GPU and toolchain identity required")
    candidate = _candidate(store, generation, request["run_id"], measurement["candidate_id"])
    _require(measurement.get("artifact_hash") == candidate["artifact_hash"], "Completed canonical artifact differs from archived candidate")
    return {"run_id": request["run_id"], "trial_id": request["trial_id"], "request_id": request_id,
        "evaluation_id": measurement["evaluation_id"], "trials_done": state["trials_done"],
        "repair": request["repair"], "research_round": request["research_round"],
        "candidate": candidate,
        "incumbent": {**_candidate(store, generation, request["run_id"], incumbent["candidate_id"] or "baseline"), "version": incumbent["version"]},
        "canonical_files_sha256": {name: cold._sha(directory / name, generation) for name in (*completion["files"], "completion.json")},
        "hardware_identity": {key: environment[key] for key in identity_keys},
        "contract_hash": request["contract_hash"], "environment_fingerprint": request["environment_fingerprint"], **documentary}


def _records(generation):
    records = {}
    for path in sorted((generation / NAMESPACE).glob("*.json")):
        if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            continue
        marker = cold._read(path, generation)
        _require(marker.get("kind") == KIND and marker.get("schema_version") == 1
                 and re.fullmatch(r"[0-9a-f]{32}", str(marker.get("epoch_id", ""))), "Malformed confirmation recovery marker")
        commit = cold._read(generation / NAMESPACE / ("committed-" + marker["epoch_id"] + ".json"), generation)
        _require(commit.get("marker_sha256") == cold._sha(path, generation), "Confirmation recovery is not committed")
        _, request, _, _ = _record(generation, marker["canonical_request_id"])
        _require(request == marker["request"], "Completed canonical request changed after recovery")
        records[path.stem] = marker, cold._sha(path, generation)
    return records


def recovery_input_identity(generation):
    generation = Path(generation)
    records = _records(generation)
    if not records:
        _require(not (generation / NAMESPACE).exists(), "Incomplete confirmation authorization cannot run")
        return cold.recovery_input_identity(generation)
    _require(not cold._recovery_records(generation), "Mixed recovery protocols require a separate authenticated migration")
    return cold._input_identity_from_records(generation, records, namespace=NAMESPACE,
        preparation_kind=PREPARATION, canonical_request_key="canonical_request_id")


def _tip(generation):
    identity = recovery_input_identity(generation)
    records = _records(generation)
    return records[identity.tip["request_id"]][0] if records else None


def _intent_path(orchestrator, task_id, attempt):
    return orchestrator.directory / "confirmation-recovery-intents" / (hashlib.sha256(task_id.encode()).hexdigest() + "-%03d.json" % attempt)


def _payload(audit):
    return {**audit["original_task"]["payload"], "source_recovery": {"kind": KIND, "epoch_id": audit["epoch_id"],
        "marker_path": str(Path(audit["new_generation"]) / NAMESPACE / (audit["request_id"] + ".json")),
        "preparation_path": audit["preparation_path"], "preparation_sha256": audit["preparation_sha256"]}}


def _finish(orchestrator, audit, path):
    result = cold._finish(orchestrator, audit, path, namespace=NAMESPACE, payload_factory=_payload, snapshot_key=SNAPSHOT_KEY)
    return {**result, "confirmation_record_id": audit["request_id"], "previous_request_id": audit["marker"]["canonical_request_id"]}


def prepare_confirmation_recovery(orchestrator, task_id, request_id, reason):
    orchestrator._require_active_study()
    _require(isinstance(reason, str) and len(reason.strip()) >= 20, "Describe the interrupted confirmation in at least twenty characters")
    with orchestrator.journal.controller(), cold._locked(orchestrator, task_id):
        task = orchestrator.journal.task(task_id)
        _require(task and task["kind"] == "generation" and task["state"] in {"running", "recovery_required"}
                 and task["payload"]["mode"] not in {"true_upstream", "native_codex_vanilla"}, "Only an incomplete improved-agent generation can recover confirmation")
        path = _intent_path(orchestrator, task_id, task["attempt"] + 1)
        _require(not path.exists(), "Preparation already exists; resume it explicitly")
        from .suite_orchestrator import config_from_dict, _local_session_job
        config = config_from_dict(task["payload"]["config"])
        generation = orchestrator._directory(task) / "generation"
        _require(cold._read(generation / "config.json", generation) == json.loads(json.dumps(config.to_dict())), "Campaign configuration changed")
        files = cold._inventory(generation)
        projection = checkpoint_projection(generation, request_id)
        _require(projection["run_id"] == task_id == config.run_id, "Task differs from completed canonical campaign")
        models = cold._closed_model_receipts(generation)
        backend = orchestrator._backend(task, config, orchestrator._directory(task))
        terminal = backend.attach_existing()
        owned = _local_session_job(task, backend)
        _require(owned and owned == str(task.get("job_id")) == str(terminal.get("job_id"))
                 and terminal.get("terminal") is True and terminal.get("state") in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"},
                 "Previous worker must have authenticated terminal ownership")
        _require(cold._inventory(generation) == files, "Cold campaign changed during trusted inspection")
        canonical, request, completion, _ = _record(generation, request_id)
        epoch = uuid.uuid4().hex
        record_id = cold._digest({"canonical_request_id": request_id, "epoch_id": epoch})
        destination = orchestrator._directory({**task, "attempt": task["attempt"] + 1}) / "generation"
        _require(not destination.exists(), "New attempt destination is occupied")
        _require(not cold._recovery_records(generation), "Mixed recovery protocols require a separate authenticated migration")
        previous = recovery_input_identity(generation)
        marker = {"schema_version": 1, "kind": KIND, "epoch_id": epoch, "canonical_request_id": request_id,
            "request": request, "previous_attempt_id": completion["attempt_id"], "projection": projection,
            "previous_generation": str(generation), "previous_task_attempt": task["attempt"], "new_task_attempt": task["attempt"] + 1,
            "previous_session": task["session"], "previous_job_id": owned, "new_session": "suite-gen-" + secrets.token_hex(16),
            "previous_worker_observation": terminal, "completed_cli_receipts": models,
            "old_request_files": projection["canonical_files_sha256"],
            "input_identity": cold._input_capsule(generation, destination, request, files, previous_identity=previous, resolve_previous=False),
            "timing_policy": "full_new_confirmation_epoch_no_historical_pair_pool", "canonical_policy": "reuse_completed_unchanged"}
        audit = {"schema_version": 1, "kind": PREPARATION, "epoch_id": epoch, "request_id": record_id,
            "reason": reason, "created_at_unix": time.time(), "original_task": task,
            "new_task_attempt": marker["new_task_attempt"], "new_session": marker["new_session"],
            "old_generation": str(generation), "new_generation": str(destination), "source_files_sha256": files,
            "marker": marker, "preparation_path": str(path), "rng_or_checkpoint_values_decoded": True,
            "checkpoint_public_metadata_projected": True, "rng_values_projected": False}
        cold._write(path, audit)
        return _finish(orchestrator, audit, path)


def resume_confirmation_preparation(orchestrator, task_id):
    orchestrator._require_active_study()
    with orchestrator.journal.controller(), cold._locked(orchestrator, task_id):
        task = orchestrator.journal.task(task_id)
        _require(task is not None, "Unknown confirmation recovery task")
        paths = [_intent_path(orchestrator, task_id, n) for n in (task["attempt"], task["attempt"] + 1)]
        candidates = [p for p in paths if p.exists() and task["session"] in
            {cold._read(p, orchestrator.directory)["original_task"]["session"], cold._read(p, orchestrator.directory)["new_session"]}]
        _require(len(candidates) == 1, "Missing or ambiguous confirmation preparation")
        path = candidates[0]
        audit = cold._read(path, orchestrator.directory)
        if path.with_suffix(".committed.json").exists():
            validate_task_recovery(orchestrator, task)
            return {**cold._read(path.with_suffix(".committed.json"), orchestrator.directory), "replayed": True,
                    "new_job_submitted": False, "model_calls_during_preparation": 0, "gpu_requests_during_preparation": 0}
        original = audit["original_task"]
        if task == original:
            from .suite_orchestrator import config_from_dict, _local_session_job
            backend = orchestrator._backend(original, config_from_dict(original["payload"]["config"]), orchestrator._directory(original))
            terminal = backend.attach_existing()
            _require(terminal.get("terminal") is True and str(terminal.get("job_id")) == _local_session_job(original, backend) == original["job_id"],
                     "Previous worker is no longer authenticated terminal")
        return _finish(orchestrator, audit, path)


def validate_task_recovery(orchestrator, task):
    binding = task.get("payload", {}).get("source_recovery", {})
    if binding.get("kind") != KIND:
        return
    path = _intent_path(orchestrator, task["id"], task["attempt"])
    _require(str(path) == binding.get("preparation_path") and cold._sha(path, orchestrator.directory) == binding.get("preparation_sha256"), "Confirmation preparation pin mismatch")
    audit = cold._read(path, orchestrator.directory)
    _require(task["payload"] == _payload({**audit, "preparation_sha256": binding["preparation_sha256"]})
             and task["session"] == audit["new_session"] and task["attempt"] == audit["new_task_attempt"]
             and str(orchestrator._directory(task) / "generation") == audit["new_generation"], "Confirmation task binding changed")
    generation = Path(audit["new_generation"])
    marker_path = generation / NAMESPACE / (audit["request_id"] + ".json")
    _require(str(marker_path) == binding["marker_path"] and cold._read(marker_path, generation) == audit["marker"], "Confirmation marker changed")
    commit = cold._read(generation / NAMESPACE / ("committed-" + audit["epoch_id"] + ".json"), generation)
    _require(commit == cold._read(path.with_suffix(".committed.json"), orchestrator.directory)
             and commit["preparation_sha256"] == binding["preparation_sha256"], "Confirmation commit pin mismatch")
    _require(cold._read(path.with_suffix(".ready.json"), orchestrator.directory) == cold._ready_record(
        audit, binding["preparation_sha256"], cold._sha(marker_path, generation), namespace=NAMESPACE), "Confirmation readiness changed")
    recovery_input_identity(generation)


def preparation_pending(orchestrator, task):
    if task.get("kind") != "generation":
        return False
    return any((path := _intent_path(orchestrator, task["id"], attempt)).exists()
        and not path.with_suffix(".committed.json").exists() for attempt in (task["attempt"], task["attempt"] + 1))


def bind_checkpoint(runtime, snapshot):
    """Admission before graph.stream can reach another model or evaluation."""
    marker = _tip(runtime.directory)
    if marker is None:
        return
    projection = marker["projection"]
    completed = runtime.directory / "trials" / projection["trial_id"] / "result.json"
    if completed.exists():
        result = cold._read(completed, runtime.directory)
        confirmation_path = completed.parent / "evaluation-requests" / projection["request_id"] / "confirmation.json"
        confirmation = cold._read(confirmation_path, runtime.directory)
        epoch_result = cold._read(runtime.directory / NAMESPACE / ("epoch-" + marker["epoch_id"]) / "result.json", runtime.directory)
        _require(result.get("tell", {}).get("evaluation_id") == projection["evaluation_id"]
                 and result["tell"].get("state", {}).get("artifact_hash") == projection["candidate"]["artifact_hash"]
                 and result.get("evaluation_evidence", {}).get("confirmation", {}).get("sha256") == cold._sha(confirmation_path, runtime.directory)
                 and confirmation.get("measurement_evaluation_id") == projection["evaluation_id"]
                 and confirmation.get("evaluation_request_id") == projection["request_id"]
                 and confirmation.get("incumbent_version") == projection["incumbent"]["version"]
                 and confirmation.get("confirmation_recovery") == epoch_result["intent"]
                 and epoch_result["intent"].get("epoch_id") == marker["epoch_id"]
                 and all(confirmation.get(key) == value for key, value in epoch_result["confirmation"].items()),
                 "Recovered trial result is not bound to its canonical confirmation epoch")
        if tuple(snapshot.next) != ("integrate",):
            with runtime.store.connection() as db:
                claim = db.execute("SELECT trial_id FROM reward_claims WHERE run_id=? AND evaluation_id=?",
                                   (runtime.config.run_id, projection["evaluation_id"])).fetchone()
            _require(claim and claim["trial_id"] == projection["trial_id"], "Graph advanced without the recovered canonical reward")
        return
    _require(_projection(runtime.directory, projection["request_id"], snapshot, runtime.store) == projection,
             "Recovered checkpoint or incumbent changed")


def _check_result(runtime, measurement, candidate, incumbent, result):
    _require(result.get("ok") is True and result.get("status") in {"promoted", "not_improved", "inconclusive"}
             and result.get("candidate_id") == candidate.candidate_id and result.get("incumbent_id") == incumbent.candidate_id
             and result.get("artifact_hash") == candidate.artifact_hash
             and result.get("incumbent_artifact_hash") == incumbent.artifact_hash
             and result.get("contract_hash") == runtime.contract_hash,
             "New confirmation did not produce a complete bound scientific result")
    protocol = json.loads(json.dumps(asdict(runtime.evaluator.contract)))
    cases = list(measurement["cases"])
    _require(result.get("protocol") == protocol and result.get("case_ids") == cases
             and len(result.get("pairs", [])) == len(cases) * protocol["confirmation_batches"],
             "New confirmation is partial or its fixed protocol changed")
    for offset, pair in enumerate(result["pairs"]):
        batch, case = divmod(offset, len(cases))
        order = [incumbent.candidate_id, candidate.candidate_id] if batch % 2 == 0 else [candidate.candidate_id, incumbent.candidate_id]
        _require(pair.get("series") == batch and pair.get("case_id") == cases[case] and pair.get("order") == order
                 and set(pair.get("measurements", {})) == set(order), "New confirmation pair order or identities changed")


def confirmation(runtime, state, candidate, incumbent, measurement, measurement_path):
    """Return one durable complete epoch, or None for ordinary later trials."""
    marker = _tip(runtime.directory)
    if marker is None or state.get("trial_id") != marker["projection"]["trial_id"]:
        return None
    projection = marker["projection"]
    _require(measurement_path.parent.name == projection["request_id"]
             and measurement.get("evaluation_id") == projection["evaluation_id"]
             and candidate.artifact_hash == projection["candidate"]["artifact_hash"]
             and incumbent.artifact_hash == projection["incumbent"]["artifact_hash"], "Recovered confirmation artifacts or canonical identity changed")
    current = runtime.store.incumbent(runtime.config.run_id)
    _require((current["candidate_id"] or "baseline") == projection["incumbent"]["candidate_id"]
             and current["version"] == projection["incumbent"]["version"], "Recovered confirmation incumbent version changed")
    _require(runtime.environment_hash == projection["environment_fingerprint"] and runtime.contract_hash == projection["contract_hash"], "Recovered confirmation environment or contract changed")
    _record(runtime.directory, projection["request_id"])
    _require(all(cold._sha(measurement_path.parent / name, runtime.directory) == sha for name, sha in projection["canonical_files_sha256"].items()), "Completed canonical evidence changed")
    epoch_dir = runtime.directory / NAMESPACE / ("epoch-" + marker["epoch_id"])
    intent = {"schema_version": 1, "kind": "full_independent_confirmation_epoch", "epoch_id": marker["epoch_id"],
        "canonical_evaluation_id": projection["evaluation_id"], "canonical_request_id": projection["request_id"],
        "candidate_artifact_hash": candidate.artifact_hash, "incumbent_artifact_hash": incumbent.artifact_hash,
        "incumbent_version": current["version"], "historical_pairs_used": 0}
    result_path = epoch_dir / "result.json"
    cold._write(epoch_dir / "intent.json", intent)
    if result_path.exists():
        wrapped = cold._read(result_path, runtime.directory)
        _require(wrapped.get("intent") == intent and wrapped.get("confirmation", {}).get("confirmation_recovery") == intent,
                 "Confirmation epoch result binding changed")
        _check_result(runtime, measurement, candidate, incumbent, wrapped["confirmation"])
        cold._write(result_path, wrapped)  # Complete a prior publication's failed parent fsync.
        return wrapped["confirmation"]
    started_path = epoch_dir / "started.json"
    _require(not started_path.exists(), "Confirmation epoch is unfinished; authorize a new independent attempt after its worker terminates")
    cold._write(started_path, {"epoch_id": marker["epoch_id"], "started_at_unix": time.time()})
    result = json.loads(json.dumps(runtime.evaluator.confirm(candidate, incumbent), allow_nan=False))
    _check_result(runtime, measurement, candidate, incumbent, result)
    result["confirmation_recovery"] = intent
    cold._write(result_path, {"intent": intent, "confirmation": result})
    return result
