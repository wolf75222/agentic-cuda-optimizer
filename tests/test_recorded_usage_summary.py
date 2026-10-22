"""Pure event fixtures; no database, provider, scheduler or GPU is accessed."""
from copy import deepcopy

import pytest

from optimizer_agent.model_usage import summarize_recorded_model_usage


def event(kind="model", *, call="call", reservation="reservation", usage=None, event_id=None):
    value = {"kind": kind, "payload": {"call_id": call, "reservation_id": reservation}}
    if kind == "model":
        value["payload"]["usage"] = usage
    if event_id is not None:
        value["event_id"] = event_id
    return value


def test_empty_instrumented_scope_is_zero_but_pending_scope_is_not():
    zero = summarize_recorded_model_usage([{"kind": "checkpoint", "payload": {}}])
    assert zero["model_calls"] == zero["known_model_calls"] == zero["model_tokens"] == 0
    assert zero["known_model_tokens"] == 0 and zero["usage_complete"] is True
    pending = summarize_recorded_model_usage([], pending_model_calls=2)
    assert pending["model_calls"] is pending["model_tokens"] is pending["known_model_tokens"] is None
    assert pending["known_model_calls"] == 0 and pending["unreported_calls"] == 2
    assert pending["usage_complete"] is False and pending["cost_usd"] is None


def test_true_zero_and_cached_subset_are_not_replaced_or_added():
    zero = summarize_recorded_model_usage([event(usage={"input_tokens": 0, "output_tokens": 0})])
    assert zero["model_calls"] == 1 and zero["model_tokens"] == 0 and zero["usage_complete"] is True
    known = summarize_recorded_model_usage([event(usage={"input_tokens": 10, "output_tokens": 2,
        "cached_input_tokens": 8, "reasoning_output_tokens": 1, "cost_usd": 400})])
    assert known["model_tokens"] == 12 and known["cached_input_tokens"] == 8
    assert known["reasoning_output_tokens"] == 1 and known["cost_usd"] is None


@pytest.mark.parametrize("invalid", [None, True, -1, 1.5, "3"])
def test_partial_call_keeps_call_count_and_only_known_token_components(invalid):
    usage = summarize_recorded_model_usage([event(usage={"input_tokens": invalid, "output_tokens": 3})])
    assert usage["model_calls"] == usage["known_model_calls"] == 1
    assert usage["input_tokens"] is usage["model_tokens"] is None
    assert usage["known_model_tokens"] == usage["known_output_tokens"] == 3
    assert usage["usage_complete"] is False and usage["partial_usage_calls"] == 1


def test_mixed_known_and_unreported_usage_never_presents_partial_sum_as_total():
    usage = summarize_recorded_model_usage([
        event(call="known", reservation="r1", usage={"input_tokens": 10, "output_tokens": 2}),
        event(call="missing", reservation="r2", usage={}),
        event("model_uncertain", call="missing", reservation="r2")])
    assert usage["model_calls"] == usage["known_model_calls"] == 2
    assert usage["model_tokens"] is usage["input_tokens"] is usage["output_tokens"] is None
    assert usage["known_model_tokens"] == 12 and usage["unreported_calls"] == 0
    assert usage["matched_model_uncertain_calls"] == usage["unknown_usage_calls"] == 1


def test_duplicate_call_and_matched_uncertainty_are_one_observation():
    completed = event(usage={"input_tokens": 5, "output_tokens": 2})
    uncertain = event("model_uncertain")
    usage = summarize_recorded_model_usage([completed, deepcopy(completed), uncertain, deepcopy(uncertain)])
    assert usage["model_calls"] == usage["known_model_calls"] == usage["observed_calls"] == 1
    assert usage["model_tokens"] == 7 and usage["usage_complete"] is True
    assert usage["recorded_model_event_count"] == 2 and usage["matched_model_uncertain_calls"] == 1


def test_reservation_only_and_transitive_identity_aliases_deduplicate():
    reservation_only = event(call=None, usage={"input_tokens": 4, "output_tokens": 1})
    result = summarize_recorded_model_usage([reservation_only, deepcopy(reservation_only)])
    assert result["model_calls"] == 1 and result["model_tokens"] == 5
    aliases = summarize_recorded_model_usage([
        event(reservation="r1", usage={"input_tokens": 4, "output_tokens": 1}),
        event(reservation="r2", usage={"input_tokens": 4, "output_tokens": 1}),
        event("model_uncertain", call=None, reservation="r2")])
    assert aliases["model_calls"] == 1 and aliases["model_tokens"] == 5
    assert aliases["matched_model_uncertain_calls"] == 1


def test_conflicting_usage_same_call_is_not_summed_or_arbitrarily_selected():
    usage = summarize_recorded_model_usage([
        event(usage={"input_tokens": 10, "output_tokens": 2}),
        event(usage={"input_tokens": 11, "output_tokens": 2})])
    assert usage["model_calls"] == 1 and usage["model_tokens"] is None
    assert usage["known_input_tokens"] is None and usage["known_model_tokens"] == 2
    assert usage["output_tokens"] == 2 and usage["usage_complete"] is False
    assert usage["recorded_usage_conflicts"][0]["fields"] == {"input_tokens": [10, 11]}


def test_invalid_duplicate_does_not_compare_boolean_true_equal_to_integer_one():
    usage = summarize_recorded_model_usage([
        event(usage={"input_tokens": True, "output_tokens": 2}),
        event(usage={"input_tokens": 1, "output_tokens": 2})])
    assert usage["model_tokens"] is None and usage["known_model_tokens"] == 2
    assert usage["recorded_usage_conflicts"][0]["fields"] == {"input_tokens": [None, 1]}
    assert "input_tokens" in usage["recorded_usage_anomalies"][0]["fields"]


def test_cache_conflict_or_invalid_subset_does_not_erase_primary_total():
    usage = summarize_recorded_model_usage([
        event(usage={"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 8}),
        event(usage={"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 20})])
    assert usage["model_tokens"] == 12 and usage["usage_complete"] is True
    assert usage["cached_input_tokens"] is None
    assert usage["recorded_usage_conflicts"][0]["fields"] == {"cached_input_tokens": [8, None]}
    assert "cached_input_tokens" in usage["recorded_usage_anomalies"][0]["fields"]


def test_failed_call_and_pending_reservation_use_max_without_double_counting():
    failed = event("model_uncertain", call="failed", reservation="pending")
    usage = summarize_recorded_model_usage([
        event(usage={"input_tokens": 3, "output_tokens": 2}), failed, deepcopy(failed)], pending_model_calls=1)
    assert usage["model_calls"] is usage["model_tokens"] is None
    assert usage["known_model_calls"] == 1 and usage["known_model_tokens"] == 5
    assert usage["unmatched_model_uncertain_calls"] == usage["unreported_calls"] == 1
    assert usage["observed_calls"] == 2
    more_pending = summarize_recorded_model_usage([
        failed, event("model_uncertain", call="another", reservation="another")], pending_model_calls=3)
    assert more_pending["unreported_calls"] == 3 and more_pending["known_model_calls"] == 0
    assert more_pending["model_tokens"] is more_pending["known_model_tokens"] is None


def test_uncertain_without_transport_identity_can_use_stable_public_event_id():
    missing = event("model_uncertain", call=None, reservation=None, event_id=7)
    usage = summarize_recorded_model_usage([missing, deepcopy(missing)])
    assert usage["unreported_calls"] == 1 and usage["model_calls"] is None


def test_conflicting_cli_ids_for_one_reservation_cannot_prove_count_or_usage():
    usage = summarize_recorded_model_usage([
        event(call="c1", usage={"input_tokens": 3, "output_tokens": 2}),
        event(call="c2", usage={"input_tokens": 3, "output_tokens": 2})])
    assert usage["model_calls"] is usage["model_tokens"] is usage["known_model_tokens"] is None
    assert usage["known_model_calls"] == 1 and len(usage["recorded_identity_conflicts"]) == 1


def test_malformed_events_are_unknown_and_inputs_are_not_mutated():
    events = [None, {"kind": "model", "payload": None}, event(usage={"input_tokens": 0})]
    before = deepcopy(events)
    usage = summarize_recorded_model_usage(events)
    assert events == before and usage["model_calls"] is None
    assert usage["known_model_calls"] == 2 and usage["model_tokens"] is None
    assert usage["malformed_event_indices"] == [0] and usage["unreported_calls"] == 1


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_pending_reservations_are_strict_nonnegative_integer_metadata(value):
    with pytest.raises(ValueError, match="pending_model_calls"):
        summarize_recorded_model_usage([], pending_model_calls=value)


@pytest.mark.parametrize("value", [None, "", b"", {}])
def test_non_event_collection_does_not_silently_establish_zero(value):
    with pytest.raises(ValueError, match="events"):
        summarize_recorded_model_usage(value)
