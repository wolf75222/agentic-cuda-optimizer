"""Synthetic public receipts; no runtime, scheduler or private state is opened."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.report_study_fr import (
    RECOVERY_AUDITS, RECOVERY_ACCEPTANCE_CHECKS, RECOVERY_AUDIT_CHECKS,
    control_receipts, evaluation_recovery_panel, execution_evidence_panel,
)


AUDITS = {number: path for path, number in RECOVERY_AUDITS.items()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def reference(root, path):
    return {"path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def refresh(context):
    root, directory = context["root"], context["directory"]
    for name, value in context["public"].items():
        save(directory/name, value)
    proof = {"files": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                       for name in context["public"]}}
    save(directory/"proof-sha256.json", proof)
    save(directory/"evaluation-recovery-acceptance.json", context["authority"])
    context["audit"]["authority"] = reference(root, directory/"evaluation-recovery-acceptance.json")
    context["audit"]["proof_manifest"] = reference(root, directory/"proof-sha256.json")
    if context["number"] == "003":
        context["audit"]["jobs"] = deepcopy(context["authority"]["jobs"])
    save(root/AUDITS[context["number"]], context["audit"])


def fixture(root, number="003", *, elapsed=("00:00:35", "00:00:37")):
    """An invented authority and its narrowly public proof subset."""
    directory = root/("results/evaluation-recovery-functional-control-"+number)
    jobs = [{"phase": phase, "job_id": str(9000+phase), "session": "fake-session-"+str(phase),
             "allocated_gres": "gpu:h100:1",
             "terminal": {"job_id": str(9000+phase), "state": "COMPLETED", "exit_code": "0:0",
                          "terminal": True, "nodes": "fake-node", "elapsed": elapsed[phase-1]}}
            for phase in (1, 2)]
    public = {"uncertain-old-cost-observation.json":
              {"uncertain_preserved": True, "old_attempt_status": "running", "old_reservation_actual": None}}
    for job in jobs:
        public["control-proof/phase-"+str(job["phase"])+"-placement.json"] = {
            "job_id": job["job_id"], "session_id": job["session"], "gres": job["allocated_gres"],
            "actual_hostname": "fake-node", "device_uuid": "c"*32}
    authority = {"kind": "evaluation_recovery_acceptance", "control_method": "functional_static_no_model",
                 "excluded_from_quality_cohorts": True, "model_calls": 0,
                 "uncertain_old_work_preserved": True, "source_head": "a"*40, "helper_sha256": "b"*64,
                 "jobs": jobs, "ok": number == "003",
                 "status": "evaluation_recovery_acceptance_passed" if number == "003" else "incomplete",
                 "checks": {key: True for key in RECOVERY_ACCEPTANCE_CHECKS} if number == "003" else
                           {"old_cost_remains_uncertain": True, "preparation_submits_no_job": True}}
    if number == "003":
        replay = {"gpu_rpc_calls_before_evaluation_replay": 15, "gpu_rpc_calls_after_evaluation_replay": 15,
                  "gpu_rpc_calls_before_replay": 33, "gpu_rpc_calls_after_replay": 33,
                  "tree_rewards_before_replay": 1, "tree_rewards_after_replay": 1,
                  "model_calls": 0, "request_id": "d"*64, "previous_request_id": "e"*64,
                  "evaluation_id": "f"*32}
        public["control-proof/replay-observation.json"] = replay
        audit = {"kind": "independent_public_terminal_audit", "source": {"head": "a"*40, "helper_sha256": "b"*64},
                 "model_calls": 0, "excluded_from_quality_cohorts": True, "uuid": "c"*32,
                 "old_uncertain_cost_preserved": True, "new_request_id": "d"*64, "old_request_id": "e"*64,
                 "new_measured_evaluation_id": "f"*32, "old_measured_evaluation_id": "a"*32,
                 "replay": deepcopy(replay), "checks": {key: True for key in RECOVERY_AUDIT_CHECKS}}
    else:
        authority["error"] = "Refused: Baseline measurement contract differs at resume"
        audit = {"kind": "evaluation_recovery_functional_control_terminal_audit",
                 "status": "incomplete_contract_resume_refused", "recovery_acceptance_ok": False}
    context = {"root": root, "directory": directory, "number": number,
               "authority": authority, "public": public, "audit": audit}
    refresh(context)
    return context


def row(root, number):
    return next(value for value in control_receipts(root) if value["path"] == AUDITS[number])


def before_submission_failure(root):
    save(root/AUDITS["001"], {
        "kind": "isolated_recovery_control_pre_submission_failure", "status": "incomplete_retained",
        "returncode": 2, "recorded_owned_jobs": 0, "phase_launch_intents": 0, "model_calls": 0,
        "artifacts_retained": True, "retry_same_output": False, "runtime_source_head": "a"*40})


def test_missing_controls_are_unknown_without_claiming_a_success(tmp_path):
    values = [row(tmp_path, number) for number in ("001", "002", "003")]
    assert all(value["ok"] is None and value["model_calls"] is None
               and value["allocated_gpu_wall_seconds"] is None for value in values)
    html = evaluation_recovery_panel(values)
    assert "en attente" in html and "acquis — reprise" not in html


def test_three_separate_controls_keep_failed_history_and_scientific_scope(tmp_path):
    before_submission_failure(tmp_path)
    fixture(tmp_path, "002", elapsed=("00:00:40", "00:00:08"))
    fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    values = [row(tmp_path, number) for number in ("001", "002", "003")]
    assert [value["ok"] for value in values] == [False, False, True]
    assert [value["allocated_gpu_wall_seconds"] for value in values] == [0, 48, 72]
    success = values[-1]
    assert success["functional_acceptance"] is True and success["validation_level"] == "ROMEO — contrôle statique"
    assert success["source_head"] == "a"*40 and len(success["jobs"]) == 2
    assert success["model_calls"] == 0 and success["old_work_actual_cost"] is None
    assert "incertain" in success["old_work_cost_status"]
    assert success["final_qualified"] is success["scientific_gain_qualified"] is False
    assert success["excluded_from_quality_cohorts"] is True
    html = execution_evidence_panel(values)
    assert "72" in html and "sans modèle" in html and "hors cohortes §16" in html
    assert "Implémentation et tests locaux ont leurs reçus séparés" in html
    assert "sans garantie Slurm" in html and "001/002 conservent leurs statuts" in html
    assert all(path.read_bytes() == body for path, body in before.items())


@pytest.mark.parametrize("name", ["same_contract", "same_real_uuid_toolchain", "old_cost_remains_uncertain",
                                  "preparation_submits_no_job", "real_native_both_jobs", "replay_no_gpu_or_tell"])
def test_each_essential_authority_guard_is_required(tmp_path, name):
    context = fixture(tmp_path)
    context["authority"]["checks"][name] = False
    refresh(context)
    assert row(tmp_path, "003")["functional_acceptance"] is False


@pytest.mark.parametrize("change", ["audit_guard", "cleanup", "model", "quality", "same_job",
                                   "uuid", "two_gpus", "old_cost_zero", "replay_rpc", "replay_tell"])
def test_re_pinned_receipts_cannot_override_semantic_guards(tmp_path, change):
    context = fixture(tmp_path)
    if change == "audit_guard":
        context["audit"]["checks"]["no_final_exposure_no_model"] = False
    elif change == "cleanup":
        context["authority"]["cleanup_error"] = "incomplete cleanup"
    elif change == "model":
        context["authority"]["model_calls"] = 1
    elif change == "quality":
        context["authority"]["excluded_from_quality_cohorts"] = False
    elif change == "same_job":
        context["authority"]["jobs"][1]["job_id"] = "9001"
    elif change == "uuid":
        context["public"]["control-proof/phase-2-placement.json"]["device_uuid"] = "0"*32
    elif change == "two_gpus":
        context["authority"]["jobs"][1]["allocated_gres"] = "gpu:h100:2"
    elif change == "old_cost_zero":
        context["public"]["uncertain-old-cost-observation.json"]["old_reservation_actual"] = 0
    elif change == "replay_rpc":
        context["public"]["control-proof/replay-observation.json"]["gpu_rpc_calls_after_replay"] = 34
    else:
        context["public"]["control-proof/replay-observation.json"]["tree_rewards_after_replay"] = 2
    refresh(context)
    value = row(tmp_path, "003")
    assert value["ok"] is False and value["allocated_gpu_wall_seconds"] is None


def test_missing_elapsed_is_unknown_cost_without_rejecting_functional_verdict(tmp_path):
    fixture(tmp_path, elapsed=("00:00:35", None))
    value = row(tmp_path, "003")
    assert value["ok"] is True and value["allocated_gpu_wall_seconds"] is None
    assert "inconnu" in evaluation_recovery_panel([value])


def test_link_tampering_does_not_fall_back_to_audit_ok(tmp_path):
    context = fixture(tmp_path)
    context["audit"]["ok"] = True
    save(tmp_path/AUDITS["003"], context["audit"])
    save(context["directory"]/"evaluation-recovery-acceptance.json", {"ok": True})
    value = row(tmp_path, "003")
    assert value["ok"] is False and value["evidence_verified"] is False


@pytest.mark.parametrize("relative", ["private.sqlite", "sealed-inputs/private.json", "../private.json"])
def test_forged_authority_paths_are_rejected_before_read(tmp_path, monkeypatch, relative):
    context = fixture(tmp_path)
    context["audit"]["authority"] = {"path": relative, "sha256": "a"*64}
    save(tmp_path/AUDITS["003"], context["audit"])
    original = Path.read_bytes
    def guarded(path):
        assert path.name not in ("private.json", "private.sqlite"), "private payload must not be opened"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert row(tmp_path, "003")["ok"] is False


@pytest.mark.parametrize("role", ["authority", "audit"])
def test_symlink_receipts_are_rejected_before_read(tmp_path, monkeypatch, role):
    context = fixture(tmp_path)
    payload = tmp_path/"forbidden.json"
    save(payload, {"private": "synthetic"})
    path = context["directory"]/"evaluation-recovery-acceptance.json" if role == "authority" else tmp_path/AUDITS["003"]
    path.unlink()
    path.symlink_to(payload)
    original = Path.read_bytes
    def guarded(candidate):
        assert candidate != payload and candidate != path, "linked payload must not be opened"
        return original(candidate)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    value = row(tmp_path, "003")
    assert value["ok"] is False
    assert "Reprise d’évaluation" in evaluation_recovery_panel([value])


@pytest.mark.parametrize("shape", [[1], {"checks": [1]}, {"jobs": [1, 2]}, {"jobs": [{"job_id": []}, {}]}])
def test_malformed_nested_json_keeps_the_global_report_available(tmp_path, shape):
    context = fixture(tmp_path)
    if isinstance(shape, list):
        save(tmp_path/AUDITS["003"], shape)
    else:
        context["authority"].update(shape)
        refresh(context)
    value = row(tmp_path, "003")
    assert value["ok"] is False
    assert "Reprise d’évaluation" in execution_evidence_panel([value])


def test_proof_inventory_never_opens_nonpublic_members(tmp_path, monkeypatch):
    context = fixture(tmp_path)
    proof_path = context["directory"]/"proof-sha256.json"
    proof = json.loads(proof_path.read_text())
    proof["files"]["sealed-inputs/final-seed.json"] = "a"*64
    proof["files"]["generation/experiments.sqlite"] = "b"*64
    save(proof_path, proof)
    context["audit"]["proof_manifest"] = reference(tmp_path, proof_path)
    save(tmp_path/AUDITS["003"], context["audit"])
    original = Path.read_bytes
    def guarded(path):
        assert path.suffix != ".sqlite" and path.name != "final-seed.json"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert row(tmp_path, "003")["ok"] is True


@pytest.mark.parametrize("change", ["both_new_missing", "old_missing", "new_bad_hash", "old_bad_hash", "same_ids"])
def test_evaluation_ids_must_be_present_hashes_distinct_and_bound_to_replay(tmp_path, change):
    context = fixture(tmp_path)
    replay = context["public"]["control-proof/replay-observation.json"]
    if change == "both_new_missing":
        del context["audit"]["new_measured_evaluation_id"]
        del replay["evaluation_id"]
    elif change == "old_missing":
        del context["audit"]["old_measured_evaluation_id"]
    elif change == "new_bad_hash":
        context["audit"]["new_measured_evaluation_id"] = replay["evaluation_id"] = "f"*31
    elif change == "old_bad_hash":
        context["audit"]["old_measured_evaluation_id"] = "G"*32
    else:
        context["audit"]["old_measured_evaluation_id"] = context["audit"]["new_measured_evaluation_id"]
    refresh(context)
    value = row(tmp_path, "003")
    assert value["ok"] is False and value["functional_acceptance"] is False


@pytest.mark.parametrize("target,field,value", [
    ("replay", "tree_rewards_after_replay", True),
    ("replay", "tree_rewards_after_replay", 1.0),
    ("replay", "gpu_rpc_calls_after_evaluation_replay", 15.0),
    ("replay", "model_calls", False),
    ("replay", "model_calls", 0.0),
    ("audit", "tree_rewards_before_replay", True),
    ("audit", "tree_rewards_after_replay", 1.0),
    ("audit", "gpu_rpc_calls_before_replay", 33.0),
    ("audit", "gpu_rpc_calls_after_replay", 33.0),
])
def test_re_pinned_replay_counts_require_exact_integer_types(tmp_path, target, field, value):
    context = fixture(tmp_path)
    values = (context["public"]["control-proof/replay-observation.json"]
              if target == "replay" else context["audit"]["replay"])
    values[field] = value
    refresh(context)
    assert row(tmp_path, "003")["ok"] is False


@pytest.mark.parametrize("job_id", ["0", "00", "09001"])
def test_job_identity_is_positive_and_canonical_before_counting_allocations(tmp_path, job_id):
    context = fixture(tmp_path)
    context["authority"]["jobs"][0]["job_id"] = job_id
    context["authority"]["jobs"][0]["terminal"]["job_id"] = job_id
    context["public"]["control-proof/phase-1-placement.json"]["job_id"] = job_id
    refresh(context)
    value = row(tmp_path, "003")
    assert value["ok"] is False and value["allocated_gpu_wall_seconds"] is None
