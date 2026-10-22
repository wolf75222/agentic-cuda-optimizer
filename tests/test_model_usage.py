"""Pure usage observations: no model invocation, billing lookup or database."""
from copy import deepcopy

import pytest

from optimizer_agent.model_usage import inspect_model_usage, aggregate_model_usage


@pytest.mark.parametrize("value", [None, True, False, -1, 0.0, 1.5, "0", "12", [], {}])
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_non_integer_or_missing_primary_counter_is_unknown_without_coercion(field, value):
    usage = {"input_tokens": 4, "output_tokens": 2, field: value}
    original = deepcopy(usage)
    observed = inspect_model_usage(usage)
    assert observed[field] is None
    assert observed["model_tokens"] is None
    assert observed["usage_complete"] is False
    assert observed["known_model_tokens"] == (2 if field == "input_tokens" else 4)
    assert observed["cost_usd"] is None
    assert usage == original


@pytest.mark.parametrize("usage", [None, {}, True, [], "usage", {"cost_usd": None}])
def test_unreported_usage_has_no_invented_total_or_known_zero(usage):
    observed = inspect_model_usage(usage)
    assert observed["input_tokens"] is observed["output_tokens"] is None
    assert observed["model_tokens"] is observed["known_model_tokens"] is None
    assert observed["usage_complete"] is False


def test_provider_reported_integer_zero_is_known_and_dollar_cost_is_not_inferred():
    observed = inspect_model_usage({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0})
    assert observed["model_tokens"] == observed["known_model_tokens"] == 0
    assert observed["usage_complete"] is True
    assert observed["cost_usd"] is None


def test_subsets_are_not_added_and_a_bad_subset_does_not_erase_valid_primary_totals():
    known = inspect_model_usage({"input_tokens": 12, "output_tokens": 5,
                                 "cached_input_tokens": 9, "reasoning_output_tokens": 3})
    assert known["model_tokens"] == 17
    invalid = inspect_model_usage({"input_tokens": 12, "output_tokens": 5,
                                   "cached_input_tokens": 13, "reasoning_output_tokens": 6})
    assert invalid["model_tokens"] == 17 and invalid["usage_complete"] is True
    assert invalid["cached_input_tokens"] is invalid["reasoning_output_tokens"] is None
    assert set(invalid["invalid_fields"]) == {"cached_input_tokens", "reasoning_output_tokens"}


def test_empty_proven_scope_is_zero_but_one_unknown_call_is_not():
    empty = aggregate_model_usage([])
    unknown = aggregate_model_usage([{}])
    assert empty["observed_calls"] == 0 and empty["model_tokens"] == 0 and empty["usage_complete"] is True
    assert unknown["observed_calls"] == 1 and unknown["model_tokens"] is None
    assert unknown["known_model_tokens"] is None and unknown["unknown_usage_calls"] == 1


def test_aggregation_exposes_partial_lower_bounds_without_replacing_unknown_with_zero():
    usage = aggregate_model_usage([{"input_tokens": 10, "output_tokens": 2}, {"input_tokens": 0}])
    assert usage["input_tokens"] == 10 and usage["output_tokens"] is usage["model_tokens"] is None
    assert usage["known_input_tokens"] == 10 and usage["known_output_tokens"] == 2
    assert usage["known_model_tokens"] == 12 and usage["observed_calls"] == 2
    assert usage["complete_usage_calls"] == usage["partial_usage_calls"] == 1
    assert usage["unknown_usage_calls"] == 0 and usage["usage_complete"] is False


def test_unreported_call_retains_known_zero_component_but_not_a_zero_total():
    usage = aggregate_model_usage([{"input_tokens": 0, "output_tokens": 0}], unreported_calls=1)
    assert usage["model_tokens"] is None and usage["known_model_tokens"] == 0
    assert usage["observed_calls"] == 2 and usage["unknown_usage_calls"] == 1
    assert usage["complete_usage_calls"] == 1 and usage["cost_usd"] is None


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_unreported_count_is_strict_caller_metadata(value):
    with pytest.raises(ValueError):
        aggregate_model_usage([], unreported_calls=value)
