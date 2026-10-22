"""Strict provider usage observations, independent of limits or billing.

Missing or malformed counters are unknown. ``known_*`` fields are explicitly
partial lower bounds; ``model_tokens`` requires both primary counters. Cache and
reasoning counters are subsets and are never added to that total. These helpers
neither reject a model's proposal nor mutate the supplied provider observation.
"""
from __future__ import annotations

from collections.abc import Mapping


TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")


def inspect_model_usage(usage):
    """Describe one call's counters without coercion or an invented zero."""
    mapping = usage if isinstance(usage, Mapping) else {}
    invalid = {}
    counts = {}
    for name in TOKEN_FIELDS:
        value = mapping.get(name)
        if value is None:
            counts[name] = None
        elif type(value) is int and value >= 0:
            counts[name] = value
        else:
            counts[name] = None
            invalid[name] = "Expected a nonnegative integer, without coercion"
    if usage is not None and not isinstance(usage, Mapping):
        invalid["usage"] = "Expected an object"
    for subset, total in (("cached_input_tokens", "input_tokens"),
                          ("reasoning_output_tokens", "output_tokens")):
        if counts[subset] is not None and counts[total] is not None and counts[subset] > counts[total]:
            counts[subset] = None
            invalid[subset] = "Reported subset exceeds its primary counter"
    primary = [counts[name] for name in ("input_tokens", "output_tokens")]
    known = [value for value in primary if value is not None]
    complete = len(known) == 2
    return {**counts, "model_tokens": sum(known) if complete else None,
            "known_input_tokens": counts["input_tokens"],
            "known_output_tokens": counts["output_tokens"],
            "known_model_tokens": sum(known) if known else None,
            "usage_complete": complete,
            "unknown_fields": [name for name in ("input_tokens", "output_tokens") if counts[name] is None],
            "invalid_fields": invalid, "cost_usd": None}


def aggregate_model_usage(usages, *, unreported_calls=0):
    """Aggregate caller-deduplicated call observations, preserving uncertainty.

    ``unreported_calls`` represents additional unresolved call observations,
    not provider HTTP requests. Call identity/deduplication belongs to the
    caller. An explicitly empty scope has a known zero; an unreported call has
    unknown totals. Partial known components remain separate lower bounds.
    """
    if type(unreported_calls) is not int or unreported_calls < 0:
        raise ValueError("unreported_calls must be a nonnegative integer")
    observations = [inspect_model_usage(usage) for usage in usages]
    count = len(observations) + unreported_calls
    totals, known_totals = {}, {}
    for name in TOKEN_FIELDS:
        values = [row[name] for row in observations if row[name] is not None]
        totals[name] = sum(values) if len(values) == count else None
        known_totals["known_" + name] = sum(values) if values or count == 0 else None
    components = [known_totals[name] for name in ("known_input_tokens", "known_output_tokens")
                  if known_totals[name] is not None]
    complete = all(row["usage_complete"] for row in observations) and unreported_calls == 0
    complete_calls = sum(row["usage_complete"] for row in observations)
    partial_calls = sum(not row["usage_complete"] and row["known_model_tokens"] is not None for row in observations)
    return {**totals, **known_totals,
            "model_tokens": totals["input_tokens"] + totals["output_tokens"] if complete else None,
            "known_model_tokens": sum(components) if components else None,
            "usage_complete": complete, "observed_calls": count,
            "complete_usage_calls": complete_calls, "partial_usage_calls": partial_calls,
            "unknown_usage_calls": count - complete_calls - partial_calls,
            "unreported_calls": unreported_calls,
            "unknown_fields": [name for name in ("input_tokens", "output_tokens") if totals[name] is None],
            "invalid_fields": [{"observation_index": index, "fields": row["invalid_fields"]}
                               for index, row in enumerate(observations) if row["invalid_fields"]],
            "cost_usd": None}


def summarize_recorded_model_usage(events, *, pending_model_calls=0):
    """Summarize one instrumented runtime's already-read public events.

    ``model`` records attest a returned call, even when its token usage is
    absent. ``model_uncertain`` records alone do not establish an exact call
    count. Call IDs and reservation IDs connect duplicate records; matched
    uncertainty is not another call. Outstanding reservation counts can refer
    to the same failures, so the unresolved count is their maximum, not sum.

    An empty *instrumented* scope is known zero. This helper must not be used
    to infer zero from a legacy runtime that never recorded model events.
    No filesystem, provider, budget or billing operation is performed here.
    """
    if type(pending_model_calls) is not int or pending_model_calls < 0:
        raise ValueError("pending_model_calls must be a nonnegative integer")
    if isinstance(events, (str, bytes, Mapping)) or events is None:
        raise ValueError("events must be an iterable of runtime event objects")
    records, malformed, parents, owners = [], [], [], {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            malformed.append(index)
            continue
        kind = event.get("kind")
        if kind not in ("model", "model_uncertain"):
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        identity = {key: payload[key] for key in ("call_id", "reservation_id")
                    if isinstance(payload.get(key), str) and payload[key]}
        position = len(records)
        parents.append(position)
        records.append({"kind": kind, "payload": payload, "identity": identity,
                        "event_index": index})
        keys = list(identity.items())
        # A repeated public event ID is a duplicate when transport identities
        # are absent. Without any ID, different rows remain unresolved.
        event_id = event.get("event_id")
        if not keys and type(event_id) in (str, int):
            keys = [("event_id", (type(event_id).__name__, event_id))]
        for key in keys:
            if key in owners:
                parents[find(position)] = find(owners[key])
            else:
                owners[key] = position
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)

    usages, conflicts, anomalies, identity_conflicts = [], [], [], []
    matched_uncertain, unmatched_uncertain, model_event_count = 0, 0, 0
    for group in groups.values():
        models = [row for row in group if row["kind"] == "model"]
        uncertain = any(row["kind"] == "model_uncertain" for row in group)
        if not models:
            unmatched_uncertain += 1
            continue
        model_event_count += len(models)
        matched_uncertain += int(uncertain)
        context = {key + "s": sorted({row["identity"][key] for row in group
                                      if key in row["identity"]})
                   for key in ("call_id", "reservation_id")}
        observed = [inspect_model_usage(row["payload"].get("usage")) for row in models]
        consensus, differing = {}, {}
        for field in TOKEN_FIELDS:
            values = [row[field] for row in observed]
            consensus[field] = values[0] if all(value == values[0] for value in values) else None
            if any(value != values[0] for value in values):
                differing[field] = list(dict.fromkeys(values))
        if differing:
            conflicts.append({**context, "fields": differing})
        # One reservation that claims different CLI call IDs cannot establish
        # either an exact call count or usable token totals for that group.
        if len(context["call_ids"]) > 1:
            identity_conflicts.append(context)
            consensus = {field: None for field in TOKEN_FIELDS}
        for record, observation in zip(models, observed):
            if observation["invalid_fields"]:
                anomalies.append({**context, "event_index": record["event_index"],
                                  "fields": observation["invalid_fields"]})
        usages.append(consensus)

    unmatched = unmatched_uncertain + len(malformed)
    unresolved = max(unmatched, pending_model_calls)
    summary = aggregate_model_usage(usages, unreported_calls=unresolved)
    return {**summary,
            "model_calls": len(usages) if unresolved == 0 and not identity_conflicts else None,
            "known_model_calls": len(usages),
            "recorded_model_event_count": model_event_count,
            "matched_model_uncertain_calls": matched_uncertain,
            "unmatched_model_uncertain_calls": unmatched_uncertain,
            "pending_model_calls": pending_model_calls,
            "recorded_usage_conflicts": conflicts,
            "recorded_usage_anomalies": anomalies,
            "recorded_identity_conflicts": identity_conflicts,
            "malformed_event_indices": malformed,
            "scope": "One instrumented runtime's recorded CLI-call observations; known counters are lower bounds, unresolved counts are conservative observations, not provider HTTP calls or billing."}
