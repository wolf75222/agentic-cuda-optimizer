"""CPU receipt fixtures only: no historical DB, worker, model or final input."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from optimizer_agent.protocol_incidents import KIND, RULE, SCOPE, load_protocol_incidents
from optimizer_agent.quality_suite import summarize_suite


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def events_digest(events):
    return hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def incident_case(tmp_path):
    repo, run_id = tmp_path / "repo", "fixture.matmul.3141.no_autotuning"
    study = repo / "results/study"
    trial, candidate, evaluation, source = "0" * 26, "c" * 24, "e" * 32, "a" * 40
    result_path = study / "campaigns" / run_id / "attempts/001/generation/trials" / trial / "result.json"
    error = "research_blocked: genuinely absent documentary compatibility evidence"
    result = {"candidate_id": candidate, "error": error, "consumption": {"trials": 1, "gpu_seconds": 42},
              "tell": {"evaluation_id": evaluation, "valid": False, "purpose": "tree_reward",
                       "state": {"candidate_id": candidate, "trial_id": trial, "evaluations": []}}}
    result_pin = {"path": result_path.relative_to(repo).as_posix(), "sha256": store(result_path, result)}
    events = [
        {"event_id": 17, "kind": "research_blocked", "payload": {"error": error, "trial_id": trial}},
        {"event_id": 18, "kind": "result_ready", "payload": {"evaluation_id": evaluation, "trial_id": trial}},
        {"event_id": 19, "kind": "tree_reward", "payload": {"evaluation_id": evaluation, "trial_id": trial, "reward": 0}},
        {"event_id": 20, "kind": "branch_decision", "payload": {"author": "search_policy", "scope": "candidate",
            "state": "closed", "target_id": candidate, "reason": "repair budget exhausted", "evidence": [error]}},
        {"event_id": 21, "kind": "diagnostic", "payload": {"candidate_id": candidate, "trial_id": trial,
            "requires_targeted_research": True}},
    ]
    audit = {"kind": "independent_spec_residual_audit", "actual_incident": {
        "kind": "actual_public_search_protocol_violation", "controller_source": source,
        "trial_id": trial, "candidate_id": candidate,
        "result": {**result_pin, "evaluation_id": evaluation, "tell_valid": False,
                   "archived_state_evaluation_ids": [], "error": error},
        "events": events, "captured_event_subset_sha256": events_digest(events)}}
    audit_path = study / "protocol-incidents/evidence/spec-residual-audit-aaaaaaa.json"
    document = {"schema_version": 1, "kind": KIND, "incident_id": "fixture-incident", "rule": RULE,
        "scope": SCOPE, "run_id": run_id, "task_id": run_id, "attempt": 1, "trial_id": trial,
        "candidate_id": candidate, "evaluation_id": evaluation, "controller_source": source,
        "origin_audit": {"path": "work/live/spec-residual-audit-aaaaaaa.json", "sha256": store(audit_path, audit)},
        "trial_result": result_pin, "event_subset_sha256": events_digest(events)}
    annotation_path = study / "protocol-incidents/fixture-incident.json"
    store(annotation_path, document)
    plan = {"entries": [{"workload": "matmul", "seed": 3141, "mode": "no_autotuning",
            "config": {"run_id": run_id}, "group_id": "group"}], "modes": ["no_autotuning"], "workloads": ["matmul"]}
    generation = {"status": "generation_complete", "method_status": "completed", "method_error": None,
                  "usage": {"model_tokens": 234, "model_calls": 7}, "candidate": {"artifact_hash": "artifact"}}
    group = {"execution_complete": True, "all_modes_available": True, "candidates": {"no_autotuning": {
             "qualified": True, "status": "validated_no_gain", "artifact_hash": "artifact"}}}
    return dict(repo=repo, study=study, plan=plan, run_id=run_id, document=document, annotation_path=annotation_path,
                audit=audit, audit_path=audit_path, result=result, result_path=result_path,
                generation=generation, groups={"group": group})


def summarize(case, verified, observations):
    return summarize_suite(case["plan"], {case["run_id"]: case["generation"]}, case["groups"],
        protocol_incidents=verified, protocol_incident_observations=observations)


def load(case):
    return load_protocol_incidents(case["study"], case["plan"])


def test_verified_incident_changes_only_method_and_global_protocol_scope(incident_case):
    case = incident_case
    paths = [case[name] for name in ("annotation_path", "audit_path", "result_path")]
    before = {path: path.read_bytes() for path in paths}
    generation = deepcopy(case["generation"])
    verified, observations = load(case)
    assert observations[0]["status"] == "verified"
    summary = summarize(case, verified, observations)
    row = summary["rows"][0]
    assert row["generation_method_status"] == "failed"
    assert row["original_generation_method_status"] == "completed"
    assert row["original_generation_method_error"] is None
    assert row["generation_status"] == "generation_complete"
    assert row["model_tokens"] == 234 and row["model_calls"] == 7
    assert row["artifact_hash"] == "artifact" and row["gain"] == 1 and row["final_qualified"] is True
    assert summary["artifact_evaluations_qualified"] is True
    assert summary["original_scientific_qualified"] is True
    assert summary["method_protocols_qualified"] is False and summary["scientific_qualified"] is False
    assert summary["aggregates"]["no_autotuning"]["method_failed"] == 1
    assert summary["aggregates"]["no_autotuning"]["method_completed"] == 0
    assert case["generation"] == generation
    assert {path: path.read_bytes() for path in paths} == before


def test_absent_incidents_do_not_invent_method_failure_or_unknown_scope(incident_case):
    case = incident_case
    case["annotation_path"].unlink()
    verified, observations = load(case)
    summary = summarize(case, verified, observations)
    assert verified == {} and observations == []
    assert summary["method_protocols_qualified"] is True and summary["scientific_qualified"] is True
    assert summary["rows"][0]["generation_method_status"] == "completed"


def test_study_local_evidence_survives_a_different_controller_checkout_and_study_name(incident_case, tmp_path):
    case = incident_case
    moved = tmp_path / "standalone-study"
    shutil.copytree(case["study"], moved)
    verified, observations = load_protocol_incidents(moved, case["plan"])
    assert observations[0]["status"] == "verified" and case["run_id"] in verified
    evidence = verified[case["run_id"]][0]
    assert evidence["origin_audit"]["path"].startswith("work/live/")
    assert evidence["resolved_public_evidence"]["origin_audit"].startswith("protocol-incidents/evidence/")
    assert evidence["trial_result"]["path"].startswith("results/study/")
    assert evidence["resolved_public_evidence"]["trial_result"].startswith("campaigns/")


def test_ordinary_failed_method_with_retained_artifact_is_not_protocol_incident(incident_case):
    case = incident_case
    case["annotation_path"].unlink()
    case["generation"].update(method_status="failed", method_error="model failed normally",
        candidate_fallback={"retained": True, "reason": "method_failed"})
    verified, observations = load(case)
    summary = summarize(case, verified, observations)
    assert summary["rows"][0]["generation_method_status"] == "failed"
    assert summary["method_protocols_qualified"] is True
    assert summary["artifact_evaluations_qualified"] is True and summary["scientific_qualified"] is True


@pytest.mark.parametrize("target", ["audit_path", "result_path"])
def test_changed_pinned_bytes_are_invalid_and_global_protocol_scope_is_unknown(incident_case, target):
    case = incident_case
    with case[target].open("ab") as handle:
        handle.write(b"\n")
    verified, observations = load(case)
    assert verified == {} and observations[0]["status"] == "invalid"
    assert "SHA mismatch" in observations[0]["error"]
    summary = summarize(case, verified, observations)
    assert summary["method_protocols_qualified"] is None and summary["scientific_qualified"] is False
    assert summary["artifact_evaluations_qualified"] is True
    assert summary["rows"][0]["generation_method_status"] == "completed"


@pytest.mark.parametrize("mutation", ["trial", "evaluation", "reward", "event_order", "closure", "evaluated"])
def test_rehashed_false_incident_still_fails_the_specific_causal_rule(incident_case, mutation):
    case = incident_case
    incident = case["audit"]["actual_incident"]
    if mutation == "trial":
        incident["events"][1]["payload"]["trial_id"] = "1" * 26
    elif mutation == "evaluation":
        incident["events"][2]["payload"]["evaluation_id"] = "f" * 32
    elif mutation == "reward":
        incident["events"][2]["payload"]["reward"] = 0.1
    elif mutation == "event_order":
        incident["events"][3]["event_id"] += 1
    elif mutation == "closure":
        incident["events"][3]["payload"]["evidence"] = ["an unrelated compile failure"]
    else:
        case["result"]["tell"]["state"]["evaluations"] = [{"evaluation_id": "a real GPU evaluation"}]
        digest = store(case["result_path"], case["result"])
        case["document"]["trial_result"]["sha256"] = incident["result"]["sha256"] = digest
    digest = events_digest(incident["events"])
    incident["captured_event_subset_sha256"] = case["document"]["event_subset_sha256"] = digest
    case["document"]["origin_audit"]["sha256"] = store(case["audit_path"], case["audit"])
    store(case["annotation_path"], case["document"])
    verified, observations = load(case)
    assert verified == {} and observations[0]["status"] == "invalid"


def test_unknown_link_role_is_rejected_before_any_payload_read(incident_case, monkeypatch):
    case = incident_case
    private = case["repo"] / "private/never-read.json"
    private.parent.mkdir()
    private.write_text("fixture sentinel; not a real seed")
    case["document"]["origin_audit"]["path"] = private.relative_to(case["repo"]).as_posix()
    store(case["annotation_path"], case["document"])
    original = Path.read_bytes
    def guarded(path):
        assert path != private, "Unknown linked role was read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    verified, observations = load(case)
    assert verified == {} and "Unknown public audit receipt role" in observations[0]["error"]


def test_unknown_annotation_file_is_not_read(incident_case, monkeypatch):
    case = incident_case
    case["annotation_path"].unlink()
    unknown = case["annotation_path"].with_name("not-an-annotation.bin")
    unknown.write_bytes(b"fixture sentinel; no payload read allowed")
    original = Path.read_bytes
    def guarded(path):
        assert path != unknown, "Unknown annotation file was read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    verified, observations = load(case)
    assert verified == {} and observations[0]["status"] == "invalid"
    assert "Unknown protocol incident filename" in observations[0]["error"]


@pytest.mark.parametrize("alias", ["file_symlink", "parent_symlink", "hardlink"])
def test_public_receipt_alias_is_rejected_before_reading_target(incident_case, monkeypatch, alias):
    case = incident_case
    target = case["repo"] / "private" / case["audit_path"].name
    target.parent.mkdir()
    target.write_text("fixture sentinel; not a real seed")
    case["audit_path"].unlink()
    if alias == "file_symlink":
        case["audit_path"].symlink_to(target)
    elif alias == "parent_symlink":
        case["audit_path"].parent.rmdir()
        case["audit_path"].parent.symlink_to(target.parent, target_is_directory=True)
    else:
        os.link(target, case["audit_path"])
    original = Path.read_bytes
    def guarded(path):
        assert path != case["audit_path"] and path != target, "Aliased public receipt payload was read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    verified, observations = load(case)
    assert verified == {} and observations[0]["status"] == "invalid"


@pytest.mark.parametrize("invalid", [[], {"unknown": "schema"}])
def test_malformed_annotation_does_not_abort_status(incident_case, invalid):
    case = incident_case
    store(case["annotation_path"], invalid)
    verified, observations = load(case)
    assert verified == {} and observations[0]["status"] == "invalid"


def test_suite_status_consumes_public_annotations_without_historical_database_access(incident_case, monkeypatch):
    from optimizer_agent import suite_orchestrator as module
    case = incident_case
    class Journal:
        def tasks(self):
            return [{"id": case["run_id"], "kind": "generation", "result": case["generation"],
                "state": "completed", "session": "unchanged-session", "job_id": "fixture", "error": None},
                {"id": "group", "kind": "final", "result": case["groups"]["group"], "state": "completed",
                 "session": "unchanged-final", "job_id": "fixture-final", "error": None}]
        @contextmanager
        def connection(self):
            class PublicAttemptStub:
                def execute(self, query):
                    return []  # No sqlite file is opened by this CPU fixture.
            yield PublicAttemptStub()
    suite = object.__new__(module.SuiteOrchestrator)
    suite.directory, suite.plan, suite.journal, suite.concurrency = case["study"], case["plan"], Journal(), 2
    monkeypatch.setattr(module, "REPO", case["repo"])
    observed = suite.status()
    assert observed["summary"]["rows"][0]["generation_method_status"] == "failed"
    assert observed["summary"]["artifact_evaluations_qualified"] is True
    assert observed["summary"]["scientific_qualified"] is False


def test_two_native_probes_keep_exact_old_raw_bytes_in_content_addressed_archive(tmp_path):
    from optimizer_agent.suite_orchestrator import write_native_preflight
    latest = tmp_path / "native-read-isolation-preflight.json"
    old_raw = b'{"ok":true,"probe":"first"}\n'
    latest.write_bytes(old_raw)
    old_sha = hashlib.sha256(old_raw).hexdigest()
    second = {"ok": True, "probe": "second"}
    receipt = write_native_preflight(tmp_path, second)
    assert Path(receipt["previous"]["path"]).read_bytes() == old_raw
    assert receipt["previous"]["sha256"] == old_sha
    assert Path(receipt["previous"]["path"]).name == old_sha + ".json"
    assert latest.read_bytes() == Path(receipt["current"]["path"]).read_bytes()
    assert json.loads(latest.read_bytes()) == second
    unchanged = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert write_native_preflight(tmp_path, second)["current"] == receipt["current"]
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == unchanged


def test_tampered_native_probe_archive_refuses_to_replace_latest(tmp_path):
    from optimizer_agent.suite_orchestrator import write_native_preflight
    first = write_native_preflight(tmp_path, {"ok": True, "probe": "first"})
    latest = tmp_path / "native-read-isolation-preflight.json"
    old_raw = latest.read_bytes()
    Path(first["current"]["path"]).write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="content-address mismatch"):
        write_native_preflight(tmp_path, {"ok": True, "probe": "second"})
    assert latest.read_bytes() == old_raw
