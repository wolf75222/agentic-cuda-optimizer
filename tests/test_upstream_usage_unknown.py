"""Synthetic original-transport accounting only; no graph, model or GPU runs."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from optimizer_agent.upstream_original import OriginalCodexTransport


def record(usage, *, key="request", attempt=0, call_id="cli-call"):
    return {"key": key, "attempt": attempt, "result": {
        "call_id": call_id, "usage": usage, "payload": {"text": "synthetic"}}}


def transport(records=(), directory=None):
    instance = OriginalCodexTransport(object(), directory=directory)
    instance.records = deepcopy(list(records))
    instance.calls = [SimpleNamespace(**row["result"]) for row in records if row.get("result")]
    return instance


def inventory(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("usage,known", [
    ({}, None), (None, None), (True, None), ("malformed", None),
    ({"input_tokens": 12}, 12), ({"output_tokens": 0}, 0),
    ({"input_tokens": True, "output_tokens": 3}, 3),
    ({"input_tokens": -1, "output_tokens": 3}, 3),
    ({"input_tokens": "12", "output_tokens": 3}, 3),
])
def test_returned_unknown_counters_are_not_completed_zero_totals(usage, known):
    client = transport([record(usage)])
    before = deepcopy(client.records)
    observed = client.usage()
    assert observed["model_calls"] == observed["known_model_calls"] == 1
    assert observed["model_tokens"] is None and observed["usage_complete"] is False
    assert observed["known_model_tokens"] == known
    assert client.records == before and observed["cost_usd"] is None


def test_exact_zero_remains_known_and_subset_tokens_are_not_added():
    zero = transport([record({"input_tokens": 0, "output_tokens": 0,
                              "cached_input_tokens": 0, "reasoning_output_tokens": 0})]).usage()
    assert zero["model_tokens"] == zero["known_model_tokens"] == 0
    assert zero["model_calls"] == 1 and zero["usage_complete"] is True
    known = transport([record({"input_tokens": 10, "output_tokens": 3,
                               "cached_input_tokens": 8, "reasoning_output_tokens": 2})]).usage()
    assert known["model_tokens"] == 13 and known["cached_input_tokens"] == 8
    assert known["reasoning_output_tokens"] == 2


def test_mixed_complete_and_partial_returns_keep_only_explicit_lower_bounds():
    observed = transport([
        record({"input_tokens": 10, "output_tokens": 2}),
        record({"input_tokens": 4}, key="another", call_id="cli-another")]).usage()
    assert observed["model_calls"] == 2 and observed["model_tokens"] is None
    assert observed["input_tokens"] == 14 and observed["output_tokens"] is None
    assert observed["known_model_tokens"] == 16 and observed["known_output_tokens"] == 2


def test_journal_replay_and_calls_mirror_do_not_duplicate_consumption():
    returned = record({"input_tokens": 10, "output_tokens": 2})
    client = transport([returned, deepcopy(returned), {**returned, "attempt": 1}])
    client.calls *= 4
    observed = client.usage()
    assert observed["acknowledged_model_records"] == 3
    assert observed["recorded_model_event_count"] == 3
    assert observed["model_calls"] == observed["known_model_calls"] == 1
    assert observed["model_tokens"] == 12


def test_conflicting_duplicate_counters_are_unknown_instead_of_selected_or_summed():
    observed = transport([record({"input_tokens": 10, "output_tokens": 2}),
                          record({"input_tokens": 11, "output_tokens": 2})]).usage()
    assert observed["model_calls"] == 1 and observed["model_tokens"] is None
    assert observed["known_input_tokens"] is None and observed["known_model_tokens"] == 2
    assert observed["recorded_usage_conflicts"][0]["fields"] == {"input_tokens": [10, 11]}


def test_one_acknowledgement_claiming_distinct_cli_ids_cannot_prove_exact_count():
    observed = transport([record({"input_tokens": 10, "output_tokens": 2}),
                          record({"input_tokens": 10, "output_tokens": 2}, call_id="conflicting")]).usage()
    assert observed["model_calls"] is observed["model_tokens"] is None
    assert observed["known_model_calls"] == 1 and observed["recorded_identity_conflicts"]


def test_error_acknowledgements_do_not_prove_that_three_cli_calls_executed():
    records = [{"key": "retry", "attempt": attempt, "error": "synthetic failure", "usage": None}
               for attempt in range(3)]
    observed = transport(records).usage()
    assert observed["model_calls"] is observed["model_tokens"] is None
    assert observed["known_model_calls"] == 0 and observed["known_model_tokens"] is None
    assert observed["unmatched_model_uncertain_calls"] == 3
    assert len(observed["uncertain_model_records"]) == 3


def test_pending_intents_are_unknown_without_loading_or_reconciling_their_contents(tmp_path):
    known = record({"input_tokens": 10, "output_tokens": 2})
    failed = {"key": "failed", "attempt": 0, "error": "synthetic failure"}
    for key in ("request", "failed", "pending"):
        path = tmp_path / "intents" / key / "0" / "intent.json"
        path.parent.mkdir(parents=True)
        # Deliberately not valid JSON: usage must only enumerate identities.
        path.write_text("Not a reply; no model request may be reconstructed from this fixture")
    client = transport([known, failed], directory=tmp_path)
    before = inventory(tmp_path)
    observed = client.usage()
    assert observed["uncertain_model_intents"] == ["intents/pending/0/intent.json"]
    assert observed["pending_model_calls"] == 1 and observed["unreported_calls"] == 2
    assert observed["model_calls"] is observed["model_tokens"] is None
    assert observed["known_model_calls"] == 1 and observed["known_model_tokens"] == 12
    assert inventory(tmp_path) == before


def test_restored_journal_retains_unknown_usage_without_a_second_backend_call(tmp_path):
    records = [record({"input_tokens": 10}), record({}, key="second", call_id="second-call")]
    (tmp_path / "calls.json").write_text(json.dumps(records))
    class Backend:
        def generate_json(self, *args, **kwargs):
            pytest.fail("Accounting must never issue a model call")
    client = OriginalCodexTransport(Backend(), directory=tmp_path)
    before = inventory(tmp_path)
    observed = client.usage()
    assert observed["model_calls"] == 2 and observed["model_tokens"] is None
    assert observed["known_model_tokens"] == 10
    assert inventory(tmp_path) == before


def test_empty_instrumented_transport_is_zero_but_an_unacknowledged_intent_is_not(tmp_path):
    zero = transport().usage()
    assert zero["model_calls"] == zero["model_tokens"] == 0 and zero["usage_complete"] is True
    pending = tmp_path / "intents/request/0/intent.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{}")
    observed = transport(directory=tmp_path).usage()
    assert observed["model_calls"] is observed["model_tokens"] is None
    assert observed["known_model_calls"] == 0 and observed["known_model_tokens"] is None


def test_malformed_journal_row_is_uncertain_without_mutating_the_journal():
    client = transport()
    client.records = [None, {"key": "missing-result", "attempt": 0}]
    before = deepcopy(client.records)
    observed = client.usage()
    assert observed["model_calls"] is observed["model_tokens"] is None
    assert observed["known_model_calls"] == 0 and observed["unreported_calls"] == 2
    assert client.records == before
