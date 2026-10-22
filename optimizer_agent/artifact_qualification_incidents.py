"""Authenticate explicit artifact disqualifications from public, pinned roles.

This is a refusal registry, not a CUDA race detector or an admission oracle.
An absent incident never proves an artifact safe. Uncertain incidents must be
handled conservatively by the caller, including unidentified/global uncertainty.
No scores, rewards, historical results, inputs or private databases are changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from .compile_cache import content_hash

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\Z")
EPOCH = re.compile(r"[0-9]{6}-[0-9a-f]{32}\Z")
ISSUE = "unsynchronized_duplicate_global_writers"
IDENTITY_KEYS = {"run_id", "candidate_id", "artifact_hash", "source_sha256", "launch_rule_sha256",
    "controller_source_sha", "job_id", "session_id", "trial_id", "evaluation_request_id", "evaluation_id",
    "task_epoch_id", "controller_epoch_id"}
INCIDENT_KEYS = {"schema_version", "kind", "incident_id", "status", "issue",
    "scientific_qualification_eligible", "identity", "evidence", "scope"}
MAX_ROLE_BYTES = 8 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, name):
    _require(isinstance(value, dict), name + " must be an object")
    return value


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        _require(key not in result, "Duplicate JSON member")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def _json(raw):
    return _object(json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_constant=_bad_constant), "JSON root")


def _read_public(root, relative, limit=MAX_ROLE_BYTES):
    """Open only regular files below root; never follow a role or parent symlink."""
    path = PurePosixPath(relative)
    _require(isinstance(relative, str) and not path.is_absolute() and str(path) == relative
             and path.parts and all(p not in {".", ".."} for p in path.parts), "Unsafe public role path")
    descriptors = []
    try:
        descriptors.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        for component in path.parts[:-1]:
            descriptors.append(os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=descriptors[-1]))
        # Nonblocking open is necessary before fstat can reject a FIFO/device.
        descriptors.append(os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptors[-1]))
        before = os.fstat(descriptors[-1])
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
                 "Public role is not a bounded regular unaliased file")
        chunks, length = [], 0
        while True:
            chunk = os.read(descriptors[-1], min(65536, limit + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
            _require(length <= limit, "Public role exceeds technical read limit")
        after = os.fstat(descriptors[-1])
        _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                 (after.st_size, after.st_mtime_ns, after.st_ino), "Public role changed while read")
        return b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _paths(identity):
    run = "campaigns/" + identity["run_id"]
    generation = run + "/generation"
    trial = generation + "/trials/" + identity["trial_id"]
    request = trial + "/evaluation-requests/" + identity["evaluation_request_id"]
    return {"result": run + "/result.json", "candidate": generation + "/candidate.json",
        "best_launch": generation + "/best-launch.json", "source": generation + "/best.cu",
        "trial_result": trial + "/result.json", "request": request + "/request.json",
        "measurement": request + "/measurement.json", "completion": request + "/completion.json",
        "worker_session": run + "/worker/slurm-session.json",
        "task_epoch": "dispatch-attestations/tasks/" + _hash(identity["run_id"].encode()) + "/" + identity["task_epoch_id"] + ".json",
        "controller_epoch": "dispatch-attestations/controller/" + identity["controller_epoch_id"] + ".json",
        "review": "artifact-incidents/" + identity["artifact_hash"] + "/independent-review.json"}


def _validate_identity(value, artifact):
    value = _object(value, "identity")
    _require(set(value) == IDENTITY_KEYS, "Unknown or missing identity role")
    for key in ("run_id", "candidate_id", "trial_id", "session_id"):
        _require(isinstance(value[key], str) and NAME.fullmatch(value[key]) and value[key] not in {".", ".."}, "Unsafe " + key)
    for key in ("artifact_hash", "source_sha256", "launch_rule_sha256", "evaluation_request_id"):
        _require(isinstance(value[key], str) and HEX64.fullmatch(value[key]), "Invalid " + key)
    _require(value["artifact_hash"] == artifact, "Incident directory and artifact differ")
    _require(isinstance(value["controller_source_sha"], str) and HEX40.fullmatch(value["controller_source_sha"]), "Invalid controller source")
    _require(isinstance(value["evaluation_id"], str) and HEX32.fullmatch(value["evaluation_id"]), "Invalid evaluation ID")
    _require(isinstance(value["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", value["job_id"]), "Invalid owned job ID")
    for key in ("task_epoch_id", "controller_epoch_id"):
        _require(isinstance(value[key], str) and EPOCH.fullmatch(value[key]), "Invalid " + key)
    return value


def _epoch(value, identity, kind):
    _require(value.get("kind") == kind and value.get("epoch_id") == identity[kind + "_epoch_id"], "Epoch identity differs")
    _require(value.get("record_hash") == content_hash({k:v for k,v in value.items() if k != "record_hash"}), "Epoch record digest differs")
    git = _object(_object(value.get("source_identity"), "source identity").get("git"), "Git identity")
    _require(git.get("available") is True and git.get("HEAD_at_dispatch") == identity["controller_source_sha"], "Controller source differs")


def _launch_rule(value):
    """Check the full serialized LaunchRule, without importing execution code.

    Artifact hashing uses these exact already-frozen fields; no default is
    inferred and no launch is applied or repaired. Other rule types remain
    uncertain until their public refusal schema is deliberately supported.
    """
    value = _object(value, "launch rule")
    _require(set(value) == {"kind", "block", "grid", "elements_key", "shared_memory_bytes", "version", "tile_x", "tile_y"}
             and value["kind"] in {"elementwise", "square_2d", "fixed"}
             and type(value["version"]) is int and value["version"] == 1, "Unsupported frozen launch schema")
    for name in ("grid", "block"):
        _require(isinstance(value[name], list) and len(value[name]) == 3 and all(type(x) is int and x > 0 for x in value[name]), "Invalid frozen launch dimensions")
    _require(type(value["shared_memory_bytes"]) is int and value["shared_memory_bytes"] >= 0
             and isinstance(value["elements_key"], str) and NAME.fullmatch(value["elements_key"]), "Invalid frozen launch fields")
    for name in ("tile_x", "tile_y"):
        _require(value[name] is None or type(value[name]) is int and value[name] > 0, "Invalid frozen tile")
    return value


def _authenticate(root, incident, artifact):
    _require(set(incident) == INCIDENT_KEYS and type(incident["schema_version"]) is int and incident["schema_version"] == 1,
             "Unsupported incident schema")
    _require(incident["kind"] == "declared_artifact_qualification_incident" and incident["status"] == "confirmed_static_violation"
             and incident["issue"] == ISSUE and incident["scientific_qualification_eligible"] is False,
             "Incident is not an explicit disqualification")
    identity = _validate_identity(incident["identity"], artifact)
    _require(incident["incident_id"] == ISSUE + "-" + artifact and isinstance(incident["scope"], str)
             and incident["scope"].strip(), "Missing explicit incident scope")
    expected, evidence = _paths(identity), _object(incident["evidence"], "evidence")
    _require(set(evidence) == set(expected), "Unexpected/missing public evidence roles")
    objects, hashes = {}, {}
    for role, path in expected.items():
        pin = _object(evidence[role], role + " pin")
        _require(set(pin) == {"path", "sha256"} and pin["path"] == path and isinstance(pin["sha256"], str)
                 and HEX64.fullmatch(pin["sha256"]), "Public role path or digest differs: " + role)
        raw = _read_public(root, path)
        _require(_hash(raw) == pin["sha256"], "Pinned public evidence changed: " + role)
        hashes[role] = _hash(raw)
        objects[role] = raw if role == "source" else _json(raw)
    result, candidate, launch = (objects[key] for key in ("result", "candidate", "best_launch"))
    _require(candidate == launch, "Candidate and best launch aliases differ")
    _require(_hash(objects["source"]) == identity["source_sha256"] == candidate.get("source_hash"), "Frozen source differs")
    rule = _launch_rule(candidate.get("launch_rule"))
    parameters = _object(candidate.get("parameters"), "parameters")
    artifact_computed = content_hash({"source": identity["source_sha256"], "kernel": candidate.get("kernel"),
        "launch_rule": rule, "parameters": json.dumps(parameters, sort_keys=True, allow_nan=False)})
    _require(artifact_computed == artifact == candidate.get("artifact_hash") and content_hash(rule) == identity["launch_rule_sha256"], "Source/launch/parameters artifact differs")
    retained = _object(result.get("candidate"), "retained candidate")
    _require(result.get("run_id") == identity["run_id"] and result.get("best_candidate_id") == candidate.get("candidate_id") == identity["candidate_id"]
             and result.get("artifact_hash") == retained.get("artifact_hash") == artifact
             and retained.get("candidate_id") == identity["candidate_id"] and retained.get("source_hash") == identity["source_sha256"]
             and retained.get("launch_rule") == candidate["launch_rule"] and retained.get("parameters") == parameters,
             "Retained candidate identities differ")
    _require(result.get("status") == "generation_complete" and result.get("generation_complete") is True
             and result.get("generation_only") is True, "Historical generation authority is not complete SEARCH")
    terminal, ack = _object(result.get("worker_terminal"), "terminal"), objects["worker_session"]
    _require(result.get("job_id") == terminal.get("job_id") == ack.get("job_id") == identity["job_id"]
             and result.get("session_id") == ack.get("session_id") == identity["session_id"]
             and terminal.get("terminal") is True and terminal.get("state") == "COMPLETED" and terminal.get("exit_code") == "0:0",
             "Owned terminal/ACK differs")
    task, controller = objects["task_epoch"], objects["controller_epoch"]
    _epoch(task, identity, "task"); _epoch(controller, identity, "controller")
    binding = _object(task.get("task"), "task binding")
    _require(binding.get("task_id") == identity["run_id"] and binding.get("kind") == "generation" and binding.get("session") == identity["session_id"]
             and task.get("controller_epoch") == {"epoch_id":identity["controller_epoch_id"], **evidence["controller_epoch"]}, "Task/controller epoch binding differs")
    request, completion, measurement = (objects[key] for key in ("request", "completion", "measurement"))
    template = _object(request.get("template"), "request template")
    # Canonical evaluation-request IDs use campaign.digest, not content_hash.
    _require(_hash(json.dumps(request, sort_keys=True, default=str).encode()) == identity["evaluation_request_id"]
             and request.get("run_id") == identity["run_id"] and request.get("trial_id") == identity["trial_id"]
             and template.get("candidate_id") == identity["candidate_id"] and template.get("artifact_hash") == artifact
             and template.get("source_sha256") == identity["source_sha256"] and template.get("launch_rule") == candidate["launch_rule"]
             and template.get("parameters") == parameters and template.get("kernel") == candidate.get("kernel"), "Canonical evaluation request differs")
    files = _object(completion.get("files"), "completion files")
    _require(completion.get("request_id") == identity["evaluation_request_id"] and completion.get("trial_id") == identity["trial_id"]
             and completion.get("candidate_id") == measurement.get("candidate_id") == identity["candidate_id"]
             and completion.get("evaluation_id") == measurement.get("evaluation_id") == identity["evaluation_id"]
             and files.get("measurement.json") == hashes["measurement"] and files.get("request.json") == hashes["request"]
             and completion.get("status") == measurement.get("status") == "measured" and measurement.get("ok") is True
             and isinstance(completion.get("attempt_id"), str) and HEX32.fullmatch(completion["attempt_id"])
             and completion.get("attempt_id") == measurement.get("attempt_id")
             and measurement.get("artifact_hash") == artifact and measurement.get("contract_hash") == result.get("contract_hash") == request.get("contract_hash"), "Durable measurement completion differs")
    trial = objects["trial_result"]
    tell = _object(trial.get("tell"), "published tell")
    measurement_ref = _object(_object(trial.get("evaluation_evidence"), "evaluation evidence").get("measurement"), "measurement reference")
    generation = "campaigns/" + identity["run_id"] + "/generation/"
    _require(trial.get("candidate_id") == identity["candidate_id"] and tell.get("evaluation_id") == identity["evaluation_id"] and tell.get("valid") is True
             and measurement_ref.get("identity") == identity["evaluation_id"] and measurement_ref.get("path") == expected["measurement"][len(generation):]
             and measurement_ref.get("sha256") == hashes["measurement"], "Trial measurement binding differs")
    review = objects["review"]
    _require(review.get("kind") == "independent_manual_static_artifact_review" and review.get("review_method") == "manual_static_source_and_archived_launch"
             and review.get("identity") == identity and review.get("issue") == ISSUE
             and review.get("verdict") == "confirmed_unsynchronized_global_writers" and review.get("scientific_qualification_eligible") is False,
             "Independent static review differs")
    _require(review.get("positive_evidence") == {k:v for k,v in evidence.items() if k != "review"},
             "Independent review pins differ from incident evidence")
    reviewed = _object(review.get("archived_launches"), "reviewed launches")
    cases = _object(measurement.get("cases"), "measurement cases")
    _require(set(cases) == set(reviewed) and cases, "Review omits measured cases")
    actual, multi_cta = {}, False
    environment = _object(result.get("environment"), "environment")
    for case_id, case in cases.items():
        series = _object(case, "case").get("series")
        _require(isinstance(series, list) and series, "Measured series missing")
        first = _object(series[0], "series").get("launch")
        _require(isinstance(first, dict) and set(first) == {"grid", "block", "shared_memory_bytes"}, "Malformed native launch")
        for name in ("grid", "block"):
            _require(isinstance(first[name], list) and len(first[name]) == 3 and all(type(x) is int and x > 0 for x in first[name]), "Invalid launch dimensions")
        multi_cta |= math.prod(first["grid"]) > 1
        for entry in series:
            entry = _object(entry, "series")
            report = _object(entry.get("report"), "native report")
            compilation = _object(report.get("compilation_identity"), "compilation identity")
            native_env = _object(report.get("environment"), "native environment")
            _require(entry.get("launch") == first and report.get("ok") is True and report.get("slurm_job_id") == identity["job_id"]
                     and isinstance(compilation.get("source"), str) and _hash(compilation["source"].encode()) == identity["source_sha256"]
                     and native_env.get("uuid") == environment.get("uuid") and isinstance(environment.get("uuid"), str) and environment["uuid"], "Archived native source/launch/job differs")
        actual[case_id] = {"series":len(series), "launch":first}
    _require(multi_cta and actual == reviewed, "Review and positive multi-CTA launch evidence differ")
    return {**incident, "authentication":{"incident_evidence_verified":True,
        "scope":"Explicit declared artifact refusal only; no automatic CUDA analysis or score mutation", "public_role_sha256":hashes}}


def load_artifact_qualification_incidents(study, run_ids):
    """Return (artifact -> authenticated refusals, observations).

    Every observation carries qualification_status='not_admitted'. ``uncertain``
    observations are not permission to proceed; unresolved identity is global
    uncertainty for the caller. No incident and out-of-scope records confer no
    positive eligibility. Only the fixed public roles in ``_paths`` are read.
    """
    root = Path(study).absolute()
    _require(not isinstance(run_ids, (str, bytes)), "Requested run IDs must be an iterable of names")
    requested = set(run_ids)
    _require(all(isinstance(x, str) and NAME.fullmatch(x) and x not in {".", ".."} for x in requested), "Invalid requested run IDs")
    incidents, observations = {}, []
    registry = root / "artifact-incidents"
    if not registry.exists() and not registry.is_symlink():
        return incidents, observations
    try:
        _require(not any(p.is_symlink() for p in (root, *root.parents, registry)) and registry.is_dir(), "Unsafe incident registry")
        entries = sorted(registry.iterdir())
        _require(len(entries) <= 256, "Incident registry exceeds technical entry limit")
    except (OSError, ValueError) as error:
        return {}, [{"status":"uncertain", "qualification_status":"not_admitted", "artifact_hash":None, "error":str(error)}]
    for folder in entries:
        artifact = folder.name if HEX64.fullmatch(folder.name) else None
        incident, declared = None, None
        try:
            _require(artifact is not None and not folder.is_symlink() and folder.is_dir(), "Invalid incident directory")
            raw = _read_public(root, "artifact-incidents/" + artifact + "/incident.json", 65536)
            incident = _json(raw)
            identity = incident.get("identity")
            if isinstance(identity, dict):
                declared = identity.get("artifact_hash")
                declared = declared if isinstance(declared, str) and HEX64.fullmatch(declared) else None
            _require(isinstance(identity, dict) and isinstance(identity.get("run_id"), str), "Unresolved incident run")
            _require(declared == artifact, "Incident directory and declared artifact conflict")
            if identity["run_id"] not in requested:
                # Do not follow another run's files. This still cannot clear an
                # identical target artifact that appears in a different run.
                observations.append({"status":"uncertain", "reason":"out_of_scope", "qualification_status":"not_admitted", "artifact_hash":declared or artifact,
                                     "run_id":identity["run_id"]})
                continue
            verified = _authenticate(root, incident, artifact)
            verified["authentication"]["incident_sha256"] = _hash(raw)
            incidents.setdefault(artifact, []).append(verified)
            observations.append({"status":"confirmed", "qualification_status":"not_admitted", "artifact_hash":artifact,
                "run_id":identity["run_id"], "incident_sha256":_hash(raw)})
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
            # Never discard A in favour of B when directory and declaration
            # conflict. An unreadable/malformed declaration also cannot resolve
            # identity from its folder name alone: retain both as context while
            # making the qualification uncertainty global.
            resolved = declared if declared is not None and declared == artifact else None
            observations.append({"status":"uncertain", "qualification_status":"not_admitted", "artifact_hash":resolved,
                "directory_artifact_hash":artifact, "declared_artifact_hash":declared, "error":str(error)})
    return incidents, observations
