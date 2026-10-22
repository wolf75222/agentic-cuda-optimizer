"""CPU-only attribution regressions using temporary orchestration fixtures.

No ROMEO receipt, private input, model, CUDA runner or active database is used.
"""
from dataclasses import replace

import pytest

from optimizer_agent import campaign
from tests.test_campaign_runtime import runtime_setup, selected, tuning_request, record_measurement


def archived_context(runtime, state):
    candidate, _ = tuning_request(runtime, state)
    measurement = record_measurement(runtime, state, candidate)
    directory = runtime.trial_dir(state)
    campaign.write_json(directory / "confirmation.json", {
        "evaluation_id": "old-confirmation", "candidate_id": candidate.candidate_id,
        "status": "not_improved", "ok": True, "interval": {"gain": 0.9}})
    campaign.write_json(directory / "profile.json", {
        "ok": False, "error": "CPU fixture profile unavailable",
        "diagnostic_only": True, "ranking_eligible": False})
    return candidate, measurement


def diagnostic(runtime, state):
    update = runtime.diagnose(state)
    return campaign.read_json(runtime.directory / "diagnostics" / (update["diagnostic_id"] + ".json"))


def assert_no_current_evaluator_evidence(value):
    assert not {row["kind"] for row in value["observations"]} & {
        "evaluation", "confirmation", "profile", "typed_failure", "profile_unavailable"}
    assert not {row["kind"] for row in value["evidence_references"]} & {
        "evaluation", "confirmation", "profile", "evaluation_request", "autotuning", "tuning_constraints"}


def test_legacy_evaluation_refusal_cannot_reappear_as_current_diagnostic(runtime_setup):
    runtime, config, _, _ = runtime_setup
    state = selected(runtime)
    candidate, _ = archived_context(runtime, state)
    paths = [runtime.trial_dir(state) / name for name in (
        f"evaluation-{candidate.candidate_id}.json", "confirmation.json", "profile.json")]
    original = {path: path.read_bytes() for path in paths}
    count = len(runtime.evaluator.benchmarks)

    update = runtime.evaluate(state)
    assert update["evaluation_status"] == "legacy_evaluation_unbound"
    assert update["evaluation_id"] is None
    state.update(update)
    runtime.integrate(state)
    value = diagnostic(runtime, state)
    assert_no_current_evaluator_evidence(value)
    evidence = runtime.generation_evidence({**state, "repair": 1})
    assert "previous_evaluation" not in evidence
    assert "previous_evaluation_reference" not in evidence
    assert {path: path.read_bytes() for path in paths} == original
    assert len(runtime.evaluator.benchmarks) == count
    assert runtime.evaluator.confirmations == []
    assert not [row for row in runtime.store.events(config.run_id) if row["kind"] == "tree_reward"]


@pytest.mark.parametrize("binding", ["absent_observation", "other_identity", "other_status"])
def test_explicit_unbound_identity_does_not_attribute_old_flat_evidence(runtime_setup, binding):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    archived_context(runtime, state)
    state["repair"] = 1
    if binding == "absent_observation":
        state.update(evaluation_id=None, evaluation_status="legacy_evaluation_unbound", valid=False)
    elif binding == "other_identity":
        state["evaluation_id"] = "a-different-evaluation"
    else:
        state["evaluation_status"] = "compilation_error"
    assert_no_current_evaluator_evidence(diagnostic(runtime, state))
    evidence = runtime.generation_evidence(state)
    assert "previous_evaluation" not in evidence
    assert "previous_evaluation_reference" not in evidence


def test_explicit_no_observation_does_not_profile_a_stale_valid_flag(runtime_setup, monkeypatch):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    archived_context(runtime, state)
    (runtime.trial_dir(state) / "profile.json").unlink()
    runtime.config = replace(runtime.config, profile=True)
    state.update(evaluation_id=None, evaluation_status="infrastructure_failure", valid=True)
    requests = []

    def prohibited_request(*args, **kwargs):
        requests.append((args, kwargs))
        raise AssertionError("An unobserved proposal cannot create a GPU profile")

    monkeypatch.setattr(runtime.evaluator, "_request", prohibited_request, raising=False)
    assert_no_current_evaluator_evidence(diagnostic(runtime, state))
    assert requests == []
    assert not (runtime.trial_dir(state) / "profile.json").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_bound_current_and_legacy_checkpoints_keep_their_observation(runtime_setup, legacy):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    _, measurement = archived_context(runtime, state)
    state["repair"] = 1
    if legacy:
        state.pop("evaluation_id")
        state.pop("evaluation_status")
    value = diagnostic(runtime, state)
    observations = {row["kind"]: row for row in value["observations"]}
    assert observations["evaluation"]["value"]["evaluation_id"] == measurement["evaluation_id"]
    assert "confirmation" in observations and "profile" in observations
    evidence = runtime.generation_evidence(state)
    assert evidence["previous_evaluation"]["evaluation_id"] == measurement["evaluation_id"]


def test_explicit_absence_does_not_reuse_old_diagnostic_marker(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    archived_context(runtime, state)
    state["repair"] = 1
    legacy = {key: value for key, value in state.items() if key not in {"evaluation_id", "evaluation_status"}}
    prior = diagnostic(runtime, legacy)
    markers = {path: path.read_bytes() for path in runtime.trial_dir(state).glob("diagnostic-*.json")}
    assert any(row["kind"] == "evaluation" for row in prior["observations"])

    state.update(evaluation_id=None, evaluation_status=None, valid=False)
    current = diagnostic(runtime, state)
    assert current["diagnostic_id"] != prior["diagnostic_id"]
    assert_no_current_evaluator_evidence(current)
    assert all(path.read_bytes() == data for path, data in markers.items())


def test_explicit_absence_ignores_both_incomplete_intent_and_legacy_alias(runtime_setup, monkeypatch):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    archived_context(runtime, state)
    incomplete = runtime.trial_dir(state) / "evaluation-requests" / ("f" * 64)
    campaign.write_json(incomplete / "intent.json", {"scope": "CPU-only unfinished request fixture"})
    state.update(repair=1, evaluation_id=None, evaluation_status="infrastructure_failure", valid=False)

    def no_current_lookup(*args):
        raise AssertionError("Explicit absence must not consult an old evaluation alias")

    monkeypatch.setattr(runtime, "_evaluation_path", no_current_lookup)
    assert_no_current_evaluator_evidence(diagnostic(runtime, state))
    assert "previous_evaluation" not in runtime.generation_evidence(state)
    assert (incomplete / "intent.json").is_file()
    assert not (incomplete / "completion.json").exists()


def test_explicit_absence_retains_only_authenticated_historical_rehost_error(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    candidate, _ = tuning_request(runtime, state)
    prior = {"evaluation_id": "old-allocation-compile-error", "candidate_id": candidate.candidate_id,
        "contract_hash": "old-contract", "status": "compilation_error", "ok": False,
        "error": "CPU fixture historical compilation failure"}
    record_measurement(runtime, state, candidate, prior)
    runtime.rehost_context = {"context_id": "CPU-only-rehost-fixture", "old_proof": {
        "identity": {"contract": "old-contract", "environment": "old-environment"},
        "checkpoint": {"state": {"candidate_id": candidate.candidate_id}}}}
    state.update(repair=1, evaluation_id=None, valid=False)

    evidence = runtime.generation_evidence(state)
    historical = evidence["previous_evaluation"]["historical_context"]
    assert historical["ranking_eligible"] is False
    assert historical["previous_contract_hash"] == "old-contract"
    assert "no current performance observation" in historical["scope"]
    assert_no_current_evaluator_evidence(diagnostic(runtime, state))
