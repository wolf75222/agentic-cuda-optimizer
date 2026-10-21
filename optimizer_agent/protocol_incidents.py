"""Read additive public protocol incidents without reopening historical state.

Only the named, byte-pinned public audit/result roles are read. Hashes bind the
observed local evidence; they are not an external signature or same-UID tamper
barrier. This classifies method outcome, never rewrites scores or qualifies an
artifact, and never opens a database, request envelope or final input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


RULE = "research_blocked_told_zero_v1"
KIND = "generation_protocol_incident"
FIELDS = {"schema_version", "kind", "incident_id", "rule", "scope", "run_id", "task_id",
          "attempt", "trial_id", "candidate_id", "evaluation_id", "controller_source",
          "origin_audit", "trial_result", "event_subset_sha256"}
SCOPE = "legacy_method_classification_only_no_historical_mutation"
MAX_JSON_BYTES = 8 * 1024 * 1024  # Technical public-receipt input guard.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_json(path: Path, root: Path):
    """Reject aliases before reading an explicitly permitted JSON receipt."""
    relative = path.relative_to(root)
    for part in (root, *(root / Path(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1))):
        _require(not part.is_symlink(), "Public receipt paths must not contain symlinks")
    metadata = path.stat()
    _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
             "Public receipt must be a regular file without hardlink aliases")
    _require(metadata.st_size <= MAX_JSON_BYTES, "Public receipt exceeds the technical JSON response limit")
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(isinstance(value, dict), "Public receipt root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _pin(value):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "Invalid public receipt pin")
    _require(isinstance(value["path"], str) and _sha(value["sha256"]), "Invalid public receipt pin values")
    path = PurePosixPath(value["path"])
    _require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value["path"],
             "Public receipt pin must be a normalized repository-relative path")
    return value


def _verify(document, filename, study, run_ids):
    _require(set(document) == FIELDS, "Unexpected or missing protocol incident fields")
    _require(document["schema_version"] == 1 and type(document["schema_version"]) is int
             and document["kind"] == KIND and document["rule"] == RULE and document["scope"] == SCOPE,
             "Unsupported protocol incident schema, rule or scope")
    _require(document["incident_id"] == filename[:-5], "Protocol incident filename identity mismatch")
    run_id = document["run_id"]
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", run_id)
             and run_id in run_ids and document["task_id"] == run_id, "Unknown generation task/run identity")
    _require(type(document["attempt"]) is int and document["attempt"] >= 0,
             "Invalid historical generation attempt")
    for field, pattern in (("trial_id", r"[0-9A-Z]{26}"), ("candidate_id", r"[0-9a-f]{24}"),
                           ("evaluation_id", r"[0-9a-f]{32}"), ("controller_source", r"[0-9a-f]{40}")):
        _require(isinstance(document[field], str) and re.fullmatch(pattern, document[field]),
                 "Invalid protocol incident " + field)
    _require(_sha(document["event_subset_sha256"]), "Invalid event subset digest")
    audit_pin, result_pin = _pin(document["origin_audit"]), _pin(document["trial_result"])
    # Exact roles are checked before any linked bytes are read.
    _require(re.fullmatch(r"work/live/spec-residual-audit-[0-9a-f]{7,40}\.json", audit_pin["path"]),
             "Unknown public audit receipt role")
    trial = document["trial_id"]
    local_result = PurePosixPath("campaigns") / run_id / "attempts" / f"{document['attempt']:03d}" / "generation" / "trials" / trial / "result.json"
    recorded_parts = PurePosixPath(result_pin["path"]).parts
    _require(len(recorded_parts) == len(local_result.parts) + 2 and recorded_parts[0] == "results"
             and re.fullmatch(r"[A-Za-z0-9_.-]+", recorded_parts[1])
             and recorded_parts[2:] == local_result.parts, "Trial result path does not match task/attempt/trial")
    # Pins retain their original repository-relative provenance. Read only the
    # exact study-local copy, independently of the controller's source checkout.
    local_audit = PurePosixPath("protocol-incidents/evidence") / PurePosixPath(audit_pin["path"]).name
    audit, audit_sha = _read_json(study / local_audit, study)
    _require(audit_sha == audit_pin["sha256"], "Origin audit SHA mismatch")
    result, result_sha = _read_json(study / local_result, study)
    _require(result_sha == result_pin["sha256"], "Trial result SHA mismatch")
    _require(audit.get("kind") == "independent_spec_residual_audit", "Unsupported origin audit kind")
    incident = audit.get("actual_incident")
    _require(isinstance(incident, dict) and incident.get("kind") == "actual_public_search_protocol_violation",
             "Origin audit has no public protocol incident")
    for field in ("trial_id", "candidate_id", "controller_source"):
        _require(incident.get(field) == document[field], "Origin audit identity mismatch: " + field)
    captured_result = incident.get("result")
    _require(isinstance(captured_result, dict) and all(captured_result.get(k) == result_pin[k] for k in result_pin),
             "Origin audit trial result pin mismatch")
    _require(captured_result.get("evaluation_id") == document["evaluation_id"]
             and captured_result.get("tell_valid") is False
             and captured_result.get("archived_state_evaluation_ids") == [], "Origin audit evaluation scope mismatch")
    error = result.get("error")
    _require(isinstance(error, str) and error.startswith("research_blocked:")
             and captured_result.get("error") == error, "Incident is not the pinned documentary refusal")
    tell = result.get("tell")
    _require(isinstance(tell, dict) and tell.get("valid") is False and tell.get("purpose") == "tree_reward"
             and tell.get("evaluation_id") == document["evaluation_id"], "Result has no matching invalid tree tell")
    state = tell.get("state")
    _require(isinstance(state, dict) and result.get("candidate_id") == state.get("candidate_id") == document["candidate_id"]
             and state.get("trial_id") == trial and state.get("evaluations") == [],
             "Result candidate/trial or unevaluated state mismatch")
    events = incident.get("events")
    _require(isinstance(events, list) and len(events) == 5 and all(isinstance(e, dict) for e in events),
             "Expected exactly five captured public protocol events")
    canonical = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    _require(hashlib.sha256(canonical).hexdigest() == document["event_subset_sha256"]
             == incident.get("captured_event_subset_sha256"), "Captured public event SHA mismatch")
    _require([event.get("kind") for event in events] == ["research_blocked", "result_ready", "tree_reward", "branch_decision", "diagnostic"],
             "Captured events do not follow the documentary-refusal/tell sequence")
    numbers = [event.get("event_id") for event in events]
    _require(all(type(number) is int and number > 0 for number in numbers)
             and numbers == list(range(numbers[0], numbers[0] + 5)), "Captured event order is not consecutive")
    payloads = [event.get("payload") for event in events]
    _require(all(isinstance(payload, dict) for payload in payloads), "Captured event payload must be an object")
    blocked, ready, reward, closure, diagnostic = payloads
    _require(blocked.get("trial_id") == trial and blocked.get("error") == error, "Refusal event does not match result")
    for payload in (ready, reward):
        _require(payload.get("trial_id") == trial and payload.get("evaluation_id") == document["evaluation_id"],
                 "Tell/reward event identity mismatch")
    _require(type(reward.get("reward")) in (int, float) and reward["reward"] == 0, "Incident lacks actual zero tree reward")
    _require(closure.get("author") == "search_policy" and closure.get("scope") == "candidate"
             and closure.get("state") == "closed" and closure.get("target_id") == document["candidate_id"]
             and closure.get("reason") == "repair budget exhausted" and isinstance(closure.get("evidence"), list)
             and error in closure["evidence"], "Closure does not retain the same documentary refusal")
    _require(diagnostic.get("trial_id") == trial and diagnostic.get("candidate_id") == document["candidate_id"]
             and diagnostic.get("requires_targeted_research") is True, "Diagnostic is not linked to the same refused trial")
    verified = {key: document[key] for key in ("incident_id", "rule", "scope", "run_id", "task_id", "attempt",
                "trial_id", "candidate_id", "evaluation_id", "controller_source", "origin_audit", "trial_result", "event_subset_sha256")}
    verified["resolved_public_evidence"] = {"origin_audit": local_audit.as_posix(), "trial_result": local_result.as_posix()}
    verified["recorded_pin_path_scope"] = "Original repository-relative provenance; resolved_public_evidence names the study-local files actually read."
    return verified


def load_protocol_incidents(study, plan):
    """Return verified run annotations and explicit invalid-file observations.

    An absent directory makes no new claim. Invalid evidence never creates a
    method failure, successful execution, changed usage or artifact decision.
    """
    study = Path(study).absolute()
    directory = study / "protocol-incidents"
    if not directory.exists() and not directory.is_symlink():
        return {}, []
    observations, verified = [], {}
    if directory.is_symlink() or not directory.is_dir():
        return {}, [{"path": "protocol-incidents", "status": "invalid", "error": "Protocol incident directory must be a real directory"}]
    if any(path.is_symlink() for path in (study, *study.parents)):
        return {}, [{"path": "protocol-incidents", "status": "invalid", "error": "Study path must not contain symlinks"}]
    run_ids = {entry["config"]["run_id"] for entry in plan["entries"]}
    for path in sorted(directory.iterdir()):
        reference = {"path": "protocol-incidents/" + path.name}
        if path.name == "evidence":
            if path.is_symlink() or not path.is_dir():
                observations.append({**reference, "status": "invalid", "error": "Public evidence archive must be a real directory"})
            continue  # Only explicitly pinned audit filenames are read below.
        try:
            _require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}\.json", path.name), "Unknown protocol incident filename")
            document, digest = _read_json(path, study)
            reference["sha256"] = digest
            annotation = _verify(document, path.name, study, run_ids)
            annotation["annotation"] = dict(reference)
            verified.setdefault(annotation["run_id"], []).append(annotation)
            observations.append({**reference, "status": "verified", "incident_id": annotation["incident_id"],
                                 "run_id": annotation["run_id"], "rule": RULE})
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            observations.append({**reference, "status": "invalid", "error": str(error)})
    return verified, observations
