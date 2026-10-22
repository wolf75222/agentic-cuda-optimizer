"""Temporary CPU ledger fixtures; no real campaign, model, SSH or GPU."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from optimizer_agent.campaign import CampaignRuntime, RecordedCodex
from optimizer_agent.experiment_store import ExperimentStore


def runtime(store):
    value = object.__new__(CampaignRuntime)
    value.store, value.config = store, SimpleNamespace(run_id="synthetic")
    return value


def model_event(store, reservation, call, tokens):
    store.event("synthetic", "model", {"reservation_id": reservation, "call_id": call,
        "usage": {"input_tokens": tokens, "output_tokens": 0}})


def charge(store, reservation, tokens, run="synthetic"):
    store.reserve(run, reservation, {"model_calls": 1})
    store.reconcile(run, reservation, {"model_calls": 1, "model_tokens": tokens})


@pytest.fixture
def store(tmp_path):
    value = ExperimentStore(tmp_path / "synthetic.sqlite")
    value.initialize_campaign("synthetic", {"contract": "fixture"})
    return value


@pytest.mark.parametrize("second_tokens", [0, 20])
def test_real_recorder_crash_after_reconciliation_never_reports_prior_calls_as_exact_total(store, monkeypatch, second_tokens):
    calls = iter([SimpleNamespace(call_id="first-cli", usage={"input_tokens": 10, "output_tokens": 0}, elapsed_seconds=1),
                  SimpleNamespace(call_id="second-cli", usage={"input_tokens": second_tokens, "output_tokens": 0}, elapsed_seconds=1)])
    backend = SimpleNamespace(generate_json=lambda *args, **kwargs: next(calls))
    recorder = RecordedCodex(backend, store, "synthetic")
    recorder.generate_json("one", {}, purpose="proposal")
    event = store.event
    def fail_publication(run, kind, payload):
        if kind == "model" and payload["call_id"] == "second-cli":
            raise OSError("synthetic crash after reconcile before model event")
        return event(run, kind, payload)
    monkeypatch.setattr(store, "event", fail_publication)
    with pytest.raises(OSError, match="after reconcile"):
        recorder.generate_json("two", {}, purpose="proposal")
    before = deepcopy(store.events("synthetic"))
    ledger_before = deepcopy(store.budget_state("synthetic"))
    observed = runtime(store).manifest_usage()
    assert observed["recorded_ledger_spent"] == {"model_calls": 2, "model_tokens": 10+second_tokens}
    assert observed["usage"]["model_calls"] is observed["usage"]["model_tokens"] is None
    details = observed["model_usage_observation"]
    assert details["known_model_calls"] == 1 and details["known_model_tokens"] == 10
    assert details["unattributed_recorded_ledger_model_entries"] == {"model_calls": 1, "model_tokens": second_tokens}
    assert len(details["unattributed_recorded_ledger_model_reservations"]) == 1
    assert store.events("synthetic") == before and store.budget_state("synthetic") == ledger_before


def test_late_durable_event_closes_only_its_matching_reservation_without_replay(store):
    charge(store, "first", 10)
    charge(store, "second", 20)
    model_event(store, "first", "cli-first", 10)
    assert runtime(store).manifest_usage()["usage"]["model_tokens"] is None
    model_event(store, "second", "cli-second", 20)
    before = deepcopy(store.events("synthetic"))
    observed = runtime(store).manifest_usage()
    assert observed["usage"]["model_tokens"] == 30 and observed["usage"]["model_calls"] == 2
    assert observed["model_usage_observation"]["unattributed_recorded_ledger_model_reservations"] == []
    assert store.events("synthetic") == before


def test_same_cli_replayed_in_two_recorded_reservations_remains_one_known_call(store):
    for reservation in ("first-ack", "second-ack"):
        charge(store, reservation, 10)
        model_event(store, reservation, "same-cli", 10)
    observed = runtime(store).manifest_usage()
    assert observed["recorded_ledger_spent"] == {"model_calls": 2, "model_tokens": 20}
    assert observed["usage"]["model_calls"] == 1 and observed["usage"]["model_tokens"] == 10
    assert observed["model_usage_observation"]["unattributed_recorded_ledger_model_reservations"] == []


def test_matching_uncertain_event_does_not_duplicate_the_missing_observation(store):
    charge(store, "first", 10)
    charge(store, "failed", 20)
    model_event(store, "first", "cli-first", 10)
    store.event("synthetic", "model_uncertain", {"reservation_id": "failed", "call_id": "failed-cli"})
    observed = runtime(store).manifest_usage()
    assert observed["usage"]["model_tokens"] is observed["usage"]["model_calls"] is None
    details = observed["model_usage_observation"]
    assert details["unmatched_model_uncertain_calls"] == 1
    assert details["unattributed_recorded_ledger_model_reservations"] == []


def test_model_reservation_view_is_run_scoped_and_excludes_pending_and_gpu_charges(store):
    store.initialize_campaign("another", {"contract": "other-fixture"})
    charge(store, "other", 99, run="another")
    charge(store, "ours", 10)
    store.reserve("synthetic", "pending", {"model_calls": 1})
    store.reserve("synthetic", "gpu", {"gpu_seconds": 3})
    store.reconcile("synthetic", "gpu", {"gpu_seconds": 3})
    assert store.reconciled_model_reservations("synthetic") == [
        {"reservation_id": "ours", "actual": {"model_calls": 1, "model_tokens": 10}}]


def test_missing_legacy_reservation_binding_stays_unknown_instead_of_guessing_by_token_total(store):
    charge(store, "exact-charge", 10)
    store.event("synthetic", "model", {"call_id": "legacy-cli", "usage": {"input_tokens": 10, "output_tokens": 0}})
    observed = runtime(store).manifest_usage()
    assert observed["usage"]["model_tokens"] is observed["usage"]["model_calls"] is None
    assert observed["model_usage_observation"]["known_model_tokens"] == 10
    assert observed["model_usage_observation"]["unattributed_recorded_ledger_model_entries"]["model_tokens"] == 10


def test_empty_instrumented_ledger_is_still_known_zero(store):
    observed = runtime(store).manifest_usage()
    assert observed["usage"]["model_calls"] == observed["usage"]["model_tokens"] == 0
    assert observed["model_usage_observation"]["unattributed_recorded_ledger_model_reservations"] == []
