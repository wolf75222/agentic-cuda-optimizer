"""Portable negative authority uses public authentication, not old score flags."""
from copy import deepcopy
import json
from types import SimpleNamespace

from optimizer_agent import artifact_qualification_incidents as authority
from optimizer_agent.portable_export import _artifact_refusal_index


ARTIFACT = "a" * 64
INCIDENT = "b" * 64
REVIEW = "c" * 64


def verified():
    return {ARTIFACT:[{"authentication":{"incident_sha256":INCIDENT},
        "evidence":{"review":{"path":"artifact-incidents/" + ARTIFACT + "/independent-review.json", "sha256":REVIEW}}}]}


def writer():
    prefix = "evidence/artifact-incidents/" + ARTIFACT + "/"
    return SimpleNamespace(files={prefix + "incident.json":{"sha256":INCIDENT},
                                  prefix + "independent-review.json":{"sha256":REVIEW}})


def test_plan_and_all_campaigns_define_public_scope_and_exact_negative_pins(tmp_path, monkeypatch):
    calls = []
    observation = {"status":"confirmed", "qualification_status":"not_admitted", "artifact_hash":ARTIFACT}
    def load(root, runs):
        calls.append((root, set(runs)))
        return verified(), [deepcopy(observation)]
    monkeypatch.setattr(authority, "load_artifact_qualification_incidents", load)
    (tmp_path / "artifact-incidents").mkdir()
    docs = [(tmp_path / "plan.json", {"entries":[{"config":{"run_id":"planned"}}]}),
            (tmp_path / "campaigns" / "archived" / "result.json", {"qualified":True})]
    errors = []
    index = _artifact_refusal_index(tmp_path, docs, writer(), errors)
    assert calls == [(tmp_path, {"planned", "archived"})]
    assert not errors
    assert index["schema_version"] == 1 and index["registry_observed"] is True
    assert index["registry_present"] is True
    assert index["observations"] == [observation]
    assert index["incidents"] == [{"artifact_hash":ARTIFACT,
        "incident_member":"evidence/artifact-incidents/" + ARTIFACT + "/incident.json",
        "incident_sha256":INCIDENT,
        "review_member":"evidence/artifact-incidents/" + ARTIFACT + "/independent-review.json",
        "review_sha256":REVIEW}]


def test_absent_registry_is_explicitly_observed_empty_not_positive_qualification(tmp_path):
    index = _artifact_refusal_index(tmp_path, [], SimpleNamespace(files={}), [])
    assert index == {"schema_version":1, "kind":"exported_artifact_refusal_index",
        "registry_observed":True, "registry_present":False, "incidents":[], "observations":[]}


def test_malformed_registry_stays_uncertain_without_following_foreign_paths(tmp_path):
    target = tmp_path / "artifact-incidents" / ARTIFACT
    target.mkdir(parents=True)
    (target / "incident.json").write_text(json.dumps({"identity":{"run_id":"foreign", "artifact_hash":"d" * 64}}))
    index = _artifact_refusal_index(tmp_path, [], SimpleNamespace(files={}), [])
    assert index["incidents"] == []
    assert index["observations"][0]["status"] == "uncertain"
    assert index["observations"][0]["artifact_hash"] is None


def test_export_bytes_must_match_loader_pins_or_archive_is_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(authority, "load_artifact_qualification_incidents", lambda *args:(verified(), []))
    inventory = writer()
    inventory.files[next(iter(inventory.files))]["sha256"] = "f" * 64
    errors = []
    index = _artifact_refusal_index(tmp_path, [], inventory, errors)
    assert not index["incidents"] and len(errors) == 1
    assert index["observations"] == [{"status":"uncertain", "qualification_status":"not_admitted",
        "artifact_hash":ARTIFACT, "reason":"exported_incident_bytes_differ_from_authenticated_registry"}]
