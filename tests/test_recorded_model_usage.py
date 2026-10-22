"""Unknown provider counters never become an observed zero in the durable ledger."""
from types import SimpleNamespace

import pytest

from optimizer_agent.campaign import RecordedCodex
from optimizer_agent.experiment_store import ExperimentStore


@pytest.mark.parametrize("usage", [
    {}, {"cost_usd": None}, {"input_tokens": None, "output_tokens": 7},
    {"input_tokens": True, "output_tokens": 3}, {"input_tokens": -1, "output_tokens": 5},
    {"input_tokens": "1", "output_tokens": "2"}, {"input_tokens": 1.5, "output_tokens": 2},
    None,
])
def test_completed_call_with_unknown_usage_is_not_rejected_or_charged_as_zero(tmp_path, usage):
    store = ExperimentStore(tmp_path / "fixture.sqlite")
    store.initialize_campaign("run", {"contract": "synthetic"})
    result = SimpleNamespace(usage=usage, call_id="completed-call", elapsed_seconds=0.5, payload={"ok": True})
    backend = SimpleNamespace(generate_json=lambda *args, **kwargs: result)
    recorder = RecordedCodex(backend, store, "run")
    assert recorder.generate_json("prompt", {}, purpose="proposal") is result
    budget = store.budget_state("run")
    assert budget["limits"] == {} and budget["reserved"] == {}
    assert budget["spent"] == {"model_calls": 1}
    events = store.events("run")
    completed = [e["payload"] for e in events if e["kind"] == "model"]
    uncertain = [e["payload"] for e in events if e["kind"] == "model_uncertain"]
    assert len(completed) == len(uncertain) == 1
    assert completed[0]["usage"] == usage
    assert completed[0]["usage_observation"]["model_tokens"] is None
    assert uncertain[0]["call_id"] == completed[0]["call_id"] == "completed-call"
    assert uncertain[0]["reservation_id"] == completed[0]["reservation_id"]
    assert uncertain[0]["model_calls_observed"] == 1
    assert uncertain[0]["reason"] == "completed_without_complete_token_usage"


@pytest.mark.parametrize("input_tokens,output_tokens", [(0, 0), (0, 7), (12, 0), (12, 3)])
def test_exact_nonnegative_integer_counts_including_zero_remain_known(tmp_path, input_tokens, output_tokens):
    store = ExperimentStore(tmp_path / "fixture.sqlite")
    store.initialize_campaign("run", {"contract": "synthetic"})
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
             "cached_input_tokens": input_tokens, "cost_usd": None}
    result = SimpleNamespace(usage=usage, call_id="known-call", elapsed_seconds=0.5)
    recorder = RecordedCodex(SimpleNamespace(generate_json=lambda *args, **kwargs: result), store, "run")
    recorder.generate_json("prompt", {}, purpose="research")
    assert store.budget_state("run")["spent"] == {
        "model_calls": 1, "model_tokens": input_tokens + output_tokens}
    assert not [e for e in store.events("run") if e["kind"] == "model_uncertain"]


def test_transport_failure_retains_unknown_consumption_with_call_and_phase(tmp_path):
    store = ExperimentStore(tmp_path / "fixture.sqlite")
    store.initialize_campaign("run", {"contract": "synthetic"})
    error = RuntimeError("synthetic transport failure")
    error.call_id = "failed-call"
    def fail(*args, **kwargs):
        raise error
    recorder = RecordedCodex(SimpleNamespace(generate_json=fail), store, "run")
    with pytest.raises(RuntimeError, match="synthetic transport"):
        recorder.generate_json("prompt", {}, purpose="research")
    events = [e["payload"] for e in store.events("run") if e["kind"] == "model_uncertain"]
    assert len(events) == 1 and events[0]["call_id"] == "failed-call"
    assert events[0]["purpose"] == "research"
    assert store.budget_state("run")["spent"] == {}


@pytest.mark.parametrize("known_tokens", [0, 15])
def test_manifest_never_labels_a_partial_ledger_charge_as_total_consumption(tmp_path, known_tokens):
    from optimizer_agent.campaign import CampaignRuntime
    store = ExperimentStore(tmp_path / "fixture.sqlite")
    store.initialize_campaign("run", {"contract": "synthetic"})
    usages = iter([{"input_tokens": known_tokens, "output_tokens": 0}, {}])
    identifiers = iter(["complete-call", "unknown-call"])
    backend = SimpleNamespace(generate_json=lambda *args, **kwargs: SimpleNamespace(
        usage=next(usages), call_id=next(identifiers), elapsed_seconds=1))
    recorded = RecordedCodex(backend, store, "run")
    recorded.generate_json("one", {}, purpose="proposal")
    recorded.generate_json("two", {}, purpose="proposal")
    runtime = object.__new__(CampaignRuntime)
    runtime.config, runtime.store = SimpleNamespace(run_id="run"), store
    snapshot = runtime.manifest_usage()
    assert snapshot["recorded_ledger_spent"]["model_tokens"] == known_tokens
    assert snapshot["usage"]["model_tokens"] is None
    assert snapshot["usage"]["model_calls"] == 2
    assert snapshot["model_usage_observation"]["known_model_tokens"] == known_tokens
    assert snapshot["model_usage_observation"]["usage_complete"] is False


@pytest.mark.parametrize("pending", [False, True])
def test_manifest_empty_instrumented_scope_differs_from_interrupted_reservation(tmp_path, pending):
    from optimizer_agent.campaign import CampaignRuntime
    store = ExperimentStore(tmp_path / "fixture.sqlite")
    store.initialize_campaign("run", {"contract": "synthetic"})
    if pending:
        store.reserve("run", "model-interrupted", {"model_calls": 1})
    runtime = object.__new__(CampaignRuntime)
    runtime.config, runtime.store = SimpleNamespace(run_id="run"), store
    snapshot = runtime.manifest_usage()
    assert snapshot["usage"]["model_tokens"] == (None if pending else 0)
    assert snapshot["usage"]["model_calls"] == (None if pending else 0)


@pytest.mark.parametrize("ledger_tokens", [0, 23])
def test_orphan_ledger_counter_without_model_observations_never_proves_consumption(tmp_path, ledger_tokens):
    from optimizer_agent.campaign import CampaignRuntime
    runtime = object.__new__(CampaignRuntime)
    runtime.config = SimpleNamespace(run_id="legacy")
    runtime.store = SimpleNamespace(events=lambda run:[], budget_state=lambda run:{
        "spent":{"model_tokens":ledger_tokens}, "reserved":{}})
    snapshot = runtime.manifest_usage()
    assert snapshot["usage"]["model_tokens"] is None
    assert snapshot["usage"]["model_calls"] is None
    assert snapshot["recorded_ledger_spent"]["model_tokens"] == ledger_tokens
    assert snapshot["model_usage_observation"]["known_model_calls"] == 0
    assert snapshot["model_usage_observation"]["unattributed_recorded_ledger_model_entries"] == {"model_tokens":ledger_tokens}
