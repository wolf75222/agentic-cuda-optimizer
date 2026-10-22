"""Synthetic ZIPs only: negative authorities must precede numeric reads."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import gpu_study_reconstructed_acceptance as helper
from test_gpu_study_reconstructed_acceptance import make_bundle, parameters


def rewrite(bundle, mutation):
    path, _, selection, quiescence, _, *arrays = bundle
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name != "export-manifest.json"}
        manifest = json.loads(archive.read("export-manifest.json"))
    mutation(manifest, members)
    manifest["files"] = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                         for name, raw in members.items()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("export-manifest.json", json.dumps(manifest).encode())
    digest = helper.sha(path)
    attestation = json.loads(quiescence.read_text())
    attestation["bundle_sha256"] = digest
    quiescence.write_text(json.dumps(attestation))
    return path, digest, selection, quiescence, helper.sha(quiescence), *arrays


def add_incident(manifest, members, *, target=True, uncertain=False):
    artifact = manifest["candidates"][0]["artifact_hash"] if target else "a" * 64
    prefix = "evidence/artifact-incidents/" + artifact + "/"
    identity = {"artifact_hash": artifact, "run_id": "synthetic.public.generation"}
    evidence = {"source": {"path": "campaigns/synthetic/generation/best.cu", "sha256": "b" * 64}}
    review = {"kind": "independent_manual_static_artifact_review",
        "review_method": "manual_static_source_and_archived_launch",
        "verdict": "confirmed_unsynchronized_global_writers", "identity": identity,
        "issue": "unsynchronized_duplicate_global_writers", "scientific_qualification_eligible": False,
        "positive_evidence": dict(evidence)}
    review_raw = json.dumps(review).encode()
    review_sha = hashlib.sha256(review_raw).hexdigest()
    evidence["review"] = {"path": prefix.removeprefix("evidence/") + "independent-review.json", "sha256": review_sha}
    incident = {"kind": "declared_artifact_qualification_incident", "status": "confirmed_static_violation",
        "identity": identity, "issue": review["issue"], "scientific_qualification_eligible": False, "evidence": evidence}
    incident_raw = json.dumps(incident).encode()
    incident_sha = hashlib.sha256(incident_raw).hexdigest()
    members[prefix + "incident.json"], members[prefix + "independent-review.json"] = incident_raw, review_raw
    index = manifest["artifact_qualification_incidents"]
    index["registry_present"] = True
    index["observations"] = [{"status": "uncertain" if uncertain else "confirmed",
        "qualification_status": "not_admitted", "artifact_hash": artifact, "run_id": identity["run_id"],
        **({} if uncertain else {"incident_sha256": incident_sha})}]
    index["incidents"] = [] if uncertain else [{"artifact_hash": artifact,
        "incident_member": prefix + "incident.json", "incident_sha256": incident_sha,
        "review_member": prefix + "independent-review.json", "review_sha256": review_sha}]
    return prefix


def refuse_before_inputs(tmp_path, monkeypatch, bundle, match):
    path, digest, options = parameters(bundle)
    original = helper.zipfile.ZipFile.read
    def no_numeric_reads(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
        assert not name.endswith(".bin"), "Refused artifact must not even read a numeric payload"
        return original(archive, member, *args, **kwargs)
    monkeypatch.setattr(helper.zipfile.ZipFile, "read", no_numeric_reads)
    monkeypatch.setattr(helper.portable_export, "reconstruct_candidate",
                        lambda *args, **kwargs: pytest.fail("No materialization after artifact refusal"))
    monkeypatch.setattr(helper, "SlurmBackend", lambda *args, **kwargs: pytest.fail("No allocation after refusal"))
    output = tmp_path / "never-materialized"
    with pytest.raises(ValueError, match=match):
        helper.prepare(output, path, digest, **options)
    assert not output.exists()


def test_legacy_archive_without_index_is_not_new_qualification(tmp_path, monkeypatch):
    bundle = rewrite(make_bundle(tmp_path / "archive"), lambda manifest, _: manifest.pop("artifact_qualification_incidents"))
    refuse_before_inputs(tmp_path, monkeypatch, bundle, "refusal index")


@pytest.mark.parametrize("uncertain", [False, True])
def test_exact_refusal_or_uncertainty_overrides_old_qualified_true_before_inputs(tmp_path, monkeypatch, uncertain):
    bundle = rewrite(make_bundle(tmp_path / "archive"),
                     lambda manifest, members: add_incident(manifest, members, uncertain=uncertain))
    refuse_before_inputs(tmp_path, monkeypatch, bundle, "explicitly refused or uncertain")


def test_global_uncertainty_prevents_input_reads(tmp_path, monkeypatch):
    def mutation(manifest, members):
        manifest["artifact_qualification_incidents"].update(registry_present=True,
            observations=[{"status": "uncertain", "qualification_status": "not_admitted", "artifact_hash": None}])
    bundle = rewrite(make_bundle(tmp_path / "archive"), mutation)
    refuse_before_inputs(tmp_path, monkeypatch, bundle, "Global or unresolved")


@pytest.mark.parametrize("damage", ["not_observed", "schema_bool", "unknown_kind", "missing_presence",
    "unindexed", "unknown_file", "wrong_member", "pin_mismatch", "review_conflict", "unproven_positive"])
def test_incoherent_refusal_index_cannot_reach_any_input(tmp_path, monkeypatch, damage):
    def mutation(manifest, members):
        prefix = add_incident(manifest, members, target=False)
        index = manifest["artifact_qualification_incidents"]
        if damage == "not_observed": index["registry_observed"] = False
        elif damage == "schema_bool": index["schema_version"] = True
        elif damage == "unknown_kind": index["kind"] = "positive_safety_certificate"
        elif damage == "missing_presence": index.pop("registry_present")
        elif damage == "unindexed": index["incidents"] = []; index["observations"] = []
        elif damage == "unknown_file": members[prefix + "unresolved.json"] = b"{}"
        elif damage == "wrong_member": index["incidents"][0]["incident_member"] = "evidence/private/input.bin"
        elif damage == "pin_mismatch": index["incidents"][0]["review_sha256"] = "0" * 64
        elif damage == "review_conflict":
            # Repin the altered bytes throughout: the refusal semantics still
            # must agree, independently of inventory checksum consistency.
            review = json.loads(members[prefix + "independent-review.json"])
            review["scientific_qualification_eligible"] = True
            raw = json.dumps(review).encode(); pin = hashlib.sha256(raw).hexdigest()
            members[prefix + "independent-review.json"] = raw
            index["incidents"][0]["review_sha256"] = pin
            incident = json.loads(members[prefix + "incident.json"])
            incident["evidence"]["review"]["sha256"] = pin
            raw = json.dumps(incident).encode(); pin = hashlib.sha256(raw).hexdigest()
            members[prefix + "incident.json"] = raw
            index["incidents"][0]["incident_sha256"] = pin
            index["observations"][0]["incident_sha256"] = pin
        elif damage == "unproven_positive": index["observations"][0]["qualification_status"] = "admitted"
    bundle = rewrite(make_bundle(tmp_path / "archive"), mutation)
    refuse_before_inputs(tmp_path, monkeypatch, bundle, "[Aa]rtifact|[Cc]onfirmed|refusal")


@pytest.mark.parametrize("uncertain", [False, True])
def test_unrelated_well_bound_refusal_is_preserved_without_certifying_target(tmp_path, uncertain):
    bundle = rewrite(make_bundle(tmp_path / "archive"),
                     lambda manifest, members: add_incident(manifest, members, target=False, uncertain=uncertain))
    path, digest, options = parameters(bundle)
    _, _, _, plan = helper.prepare(tmp_path / "prepared", path, digest, **options)
    guard = plan["artifact_refusal_guard"]
    assert guard["status"] == "no_recorded_refusal_for_selected_artifact"
    assert guard["confirmed_refusals"] == (0 if uncertain else 1)
    assert guard["targeted_uncertainties"] == int(uncertain)
    assert "no general race-free" in guard["scope"]


def test_conflicting_uncertain_identity_is_global_even_for_another_selection(tmp_path, monkeypatch):
    def mutation(manifest, members):
        prefix = add_incident(manifest, members, target=False, uncertain=True)
        incident = json.loads(members[prefix + "incident.json"])
        incident["identity"]["artifact_hash"] = manifest["candidates"][0]["artifact_hash"]
        members[prefix + "incident.json"] = json.dumps(incident).encode()
    bundle = rewrite(make_bundle(tmp_path / "archive"), mutation)
    refuse_before_inputs(tmp_path, monkeypatch, bundle, "Unresolved archived")


def test_empty_observed_registry_records_negative_only_guard(tmp_path):
    path, digest, options = parameters(make_bundle(tmp_path / "archive"))
    _, _, _, plan = helper.prepare(tmp_path / "prepared", path, digest, **options)
    assert plan["artifact_refusal_guard"]["registry_present"] is False
    assert plan["artifact_refusal_guard"]["confirmed_refusals"] == 0


def test_simple_extraction_of_legacy_archive_still_creates_no_new_qualification(tmp_path):
    bundle = rewrite(make_bundle(tmp_path / "archive"), lambda manifest, _: manifest.pop("artifact_qualification_incidents"))
    path, _, selection, *_ = bundle
    candidate, _, _ = helper.portable_export.reconstruct_candidate(path, selection, tmp_path / "extracted")
    receipt = json.loads((tmp_path / "extracted/reconstruction.json").read_text())
    assert receipt["executed"] is False and candidate.artifact_hash
    assert receipt["qualification"] == "Archived qualification does not qualify a new execution environment"
