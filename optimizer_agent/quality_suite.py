"""Predeclared §16 design and inclusive, evidence-based reporting.

Generation scores are never substituted for the shared final GPU experiment.
Unknown outcomes remain unknown in aggregates, with their full denominator.
"""
from dataclasses import replace
from copy import deepcopy
import hashlib
import math
import statistics

from .campaign_config import CampaignConfig

WORKLOADS = ("reduction", "scan", "stencil", "convolution", "matmul", "normalization")
MODES = ("full", "true_upstream", "native_codex_vanilla", "no_autotuning", "no_research", "pmpp_permanent_only", "no_pmpp")
OPTIONAL_MODES = ("greedy_parent",)
TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def stable_id(prefix, *parts):
    value = ".".join((prefix, *(str(p) for p in parts)))
    return value if len(value) <= 80 else value[:55] + "." + hashlib.sha256(value.encode()).hexdigest()[:24]


def plan_suite(base: CampaignConfig, seeds=(1729, 2718, 3141), *, include_greedy=False):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Use at least two distinct nonnegative integer campaign seeds")
    modes = MODES + (OPTIONAL_MODES if include_greedy else ())
    domain = tuple(dict.fromkeys((1, 17, 63) + tuple(base.search_sizes)))
    entries = []
    for workload in WORKLOADS:
        for seed in seeds:
            for mode in modes:
                updates = dict(workload=workload, seed=seed,
                    run_id=stable_id(base.run_id, workload, seed, mode),
                    search_sizes=domain, final_sizes=domain, required_sanitizers=TOOLS,
                    experimental_mode=None, research_required=True, web_search_required=True,
                    autotuning_enabled=True)
                if mode == "greedy_parent":
                    updates["experimental_mode"] = "ablation_no_tree"
                elif mode == "no_autotuning":
                    updates["autotuning_enabled"] = False
                elif mode == "no_research":
                    updates.update(experimental_mode="ablation_no_research", research_required=False, web_search_required=False)
                elif mode == "no_pmpp":
                    updates["experimental_mode"] = "ablation_no_pmpp"
                elif mode == "pmpp_permanent_only":
                    updates["experimental_mode"] = "ablation_pmpp_permanent_only"
                elif mode in ("true_upstream", "native_codex_vanilla"):
                    updates.update(experimental_mode="test", research_required=False, web_search_required=False)
                config = replace(base, **updates)
                entries.append({"workload": workload, "seed": seed, "mode": mode, "config": config.to_dict(),
                                "group_id": stable_id(base.run_id, workload, seed)})
    return {"schema_version": 2, "kind": "quality_ablation_plan", "suite_id": base.run_id,
            "entries": entries, "modes": modes, "seeds": seeds, "workloads": WORKLOADS,
            "global_resource_caps": None,
            "comparison_protocol": {
                "generation": "Same fixed iteration/repair controls where applicable; observed costs are not equalized",
                "domain": "Fixed admissible shapes; fresh sealed final inputs on those same shapes (spec4.3)",
                "final": "All generators frozen before one new exclusive GPU group; independent CPU oracle, all four sanitizers, fixed AB/BA paired bootstrap",
                "direct_pairs": "Full against original, native vanilla and each isolated ablation; permanent-only against no-PMPP; fixed before generation",
                "aggregation": "Baseline-normalized final gains; no pooling of raw timings from different leases",
                "unknowns": "Failures and missing evidence retain their denominator; no synthetic measured gains"},
            "ablation_contrasts": {
                "targeted_pmpp_consultation": {"candidate": "full", "reference": "pmpp_permanent_only",
                                               "comparison_key": "full_vs_pmpp_permanent_only"},
                "permanent_pmpp_guidance": {"candidate": "pmpp_permanent_only", "reference": "no_pmpp",
                                            "comparison_key": "pmpp_permanent_only_vs_no_pmpp"}},
            "method_scope": {"true_upstream": "Pinned original LangGraph optimizer with transport/model bridges only",
                "native_codex_vanilla": "One ordinary tools-enabled Codex prompt; no custom LangGraph optimizer",
                "pmpp_permanent_only": "Original permanent guide in diagnostics/generation; no targeted cards, book, companion or external PMPP sources",
                "greedy_parent": "Optional isolated parent-selection ablation; never an upstream substitute"}}


def summarize_suite(plan, manifests, groups=None, *, protocol_incidents=None, protocol_incident_observations=None,
                    artifact_qualification_incidents=None, artifact_incident_observations=None):
    groups = groups or {}
    protocol_incidents = protocol_incidents or {}
    protocol_incident_observations = protocol_incident_observations or []
    artifact_qualification_incidents = artifact_qualification_incidents or {}
    artifact_incident_observations = artifact_incident_observations or []

    # Inputs are already authenticated by the caller's incident loader. Do not
    # reread study files or revise original manifests while projecting them.
    def artifact_key(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    global_artifact_uncertainty = any(not isinstance(observation, dict) or
        not artifact_key(observation.get("artifact_hash")) for observation in artifact_incident_observations)
    global_artifact_uncertainty |= any(not artifact_key(artifact) and records
        for artifact, records in artifact_qualification_incidents.items())

    def qualification_issue(artifacts):
        records, observation_refs = [], []
        for artifact in set(value for value in artifacts if artifact_key(value)):
            records.extend(artifact_qualification_incidents.get(artifact, []))
            observation_refs.extend({"observation_index": index, "status": observation.get("status")}
                for index, observation in enumerate(artifact_incident_observations)
                if isinstance(observation, dict) and observation.get("artifact_hash") == artifact)
        return {"artifact_qualification_incident_ids": sorted({record["incident_id"] for record in records
                    if isinstance(record, dict) and isinstance(record.get("incident_id"), str)}),
                "artifact_incident_observation_refs": sorted(observation_refs, key=lambda row: row["observation_index"]),
                "artifact_incident_blocked": bool(records or observation_refs),
                "artifact_qualification_status": ("not_admitted_confirmed_incident" if records or any(
                    row["status"] == "confirmed" for row in observation_refs) else "not_admitted_uncertain_incident"
                    if observation_refs else "eligibility_unknown" if global_artifact_uncertainty else "no_incident_observed")}

    def unqualify(target, issue, *, row=False):
        # Preserve observed values explicitly, but do not leave disqualified
        # gains/intervals in the fields consumed by aggregates and comparisons.
        for key in ("status", "ok", "qualified", "final_qualified", "gain_basis", "gain", "lower", "upper",
                    "case_gains", "interval", "critical_cases_ok", "regression",
                    "time_to_independently_confirmed_final_gain_seconds"):
            if key in target:
                target["observed_unqualified_" + key] = deepcopy(target[key])
        target.update(issue, status="artifact_qualification_not_admitted", qualified=False)
        for key in ("gain_basis", "gain", "lower", "upper", "interval", "critical_cases_ok",
                    "time_to_independently_confirmed_final_gain_seconds"):
            if key in target:
                target[key] = None
        if "case_gains" in target:
            target["case_gains"] = {}
        if "ok" in target:
            target["ok"] = False
        if row:
            target.update(final_qualified=False, regression=False)

    def contrast_view(group_id, pair_key, contrast):
        # Detailed pairs/reports/requests remain only in the original group.
        # This is an incident-aware display projection, never a new measurement.
        fields = ("status", "ok", "qualified", "gain_basis", "critical_cases_ok", "gain", "lower", "upper",
                  "evaluation_id", "candidate_id", "incumbent_id", "artifact_hash", "candidate_artifact_hash",
                  "incumbent_artifact_hash", "left_artifact_hash", "right_artifact_hash", "contract_hash", "case_ids")

        def number(value):
            try:
                return value if type(value) in (int, float) and math.isfinite(value) else None
            except OverflowError:
                return None

        view = {}
        for field in fields:
            if field not in contrast:
                continue
            value = contrast[field]
            if field in ("ok", "qualified", "critical_cases_ok"):
                view[field] = value if type(value) is bool else None
            elif field in ("gain", "lower", "upper"):
                view[field] = number(value)
            elif field == "case_ids":
                view[field] = list(value) if isinstance(value, list) and all(isinstance(x, str) for x in value) else None
            else:
                view[field] = value if isinstance(value, str) else None
        view.update(kind="artifact_aware_method_comparison_summary", group_id=group_id,
                    pair_key=pair_key, raw_available_in_group=True)
        view.setdefault("qualified", None)  # An omitted original flag is not invented.
        if "interval" in contrast:
            interval = contrast["interval"]
            if isinstance(interval, dict):
                view["interval"] = {field: number(interval[field])
                    for field in ("gain", "lower", "upper", "confidence", "bootstrap_samples") if field in interval}
                if isinstance(interval.get("case_gains"), dict):
                    view["interval"]["case_gains"] = {case_id: {field: number(row[field])
                        for field in ("gain", "lower", "upper") if field in row}
                        for case_id, row in interval["case_gains"].items() if isinstance(case_id, str) and isinstance(row, dict)}
            else:
                view["interval"] = None
        return view

    rows, aggregates = [], {}
    for entry in plan["entries"]:
        run_id = entry["config"]["run_id"]
        generation = manifests.get(run_id, {})
        group = groups.get(entry.get("group_id"), {})
        result = group.get("candidates", {}).get(entry["mode"], {})
        interval = result.get("confirmation", {}).get("interval", {})
        usage = generation.get("usage", generation.get("budgets", {}).get("spent", {}))
        row = {k: entry[k] for k in ("workload", "seed", "mode")}
        row.update(run_id=run_id, generation_status=generation.get("status", generation.get("certification_status", "missing")),
            generation_method_status=generation.get("method_status"),
            generation_method_error=generation.get("method_error"),
            candidate_fallback=generation.get("candidate_fallback"),
            status=result.get("status", "pending_common_final" if generation else "missing"),
            gain_basis=result.get("gain_basis", "paired_measurement" if interval else None),
            gain=interval.get("gain"), lower=interval.get("lower"), upper=interval.get("upper"),
            case_gains=interval.get("case_gains", {}),
            model_tokens=usage.get("model_tokens"), model_calls=usage.get("model_calls"),
            gpu_reservation_wall_seconds=usage.get("gpu_seconds"),
            configurations=usage.get("local_configurations", usage.get("configurations")),
            wall_seconds=generation.get("elapsed_seconds"), metrics=generation.get("metrics", {}),
            artifact_hash=result.get("artifact_hash", generation.get("candidate", {}).get("artifact_hash")),
            generation_evidence=generation.get("evidence_path"), group_id=entry.get("group_id"),
            final_evidence=group.get("task_evidence_path"),
            total_generation_task_wall_seconds=generation.get("total_task_wall_seconds_including_queue"),
            time_to_independently_confirmed_final_gain_seconds=result.get("time_to_independently_confirmed_final_gain_seconds"),
            final_closure=result.get("final_closure"),
            final_terminal=group.get("worker_terminal"), final_environment=group.get("environment"))
        incidents = protocol_incidents.get(run_id, [])
        if incidents:
            row.update(original_generation_method_status=row["generation_method_status"],
                original_generation_method_error=row["generation_method_error"],
                generation_method_status="failed",
                generation_method_error="Legacy search protocol violation: a documentary research refusal was consumed as zero tree reward; historical tree, usage and artifact decisions are preserved.",
                generation_protocol_incidents=incidents)
        if result.get("status") == "validated_no_gain":
            row["gain"] = 1.0
            row["gain_basis"] = "exact_baseline_artifact"
        row["regression"] = row["gain"] is not None and row["gain"] < 1
        row["final_qualified"] = bool(result.get("qualified"))
        row["qualified"] = row["final_qualified"]
        issue = qualification_issue([row["artifact_hash"]])
        row.update(issue)
        if issue["artifact_incident_blocked"]:
            unqualify(row, issue, row=True)
        rows.append(row)
    for mode in plan.get("modes", MODES):
        selected = [r for r in rows if r["mode"] == mode]
        measured = [r["gain"] for r in selected if r["gain"] is not None and r["gain"] > 0]
        qualified = sum(r["final_qualified"] for r in selected)
        aggregates[mode] = {"planned": len(selected), "generation_received": sum(r["generation_status"] != "missing" for r in selected),
            "method_completed": sum(r["generation_method_status"] == "completed" for r in selected),
            "method_failed": sum(r["generation_method_status"] == "failed" for r in selected),
            "method_status_unknown": sum(r["generation_method_status"] is None for r in selected),
            "final_measured": len(measured), "qualified": qualified, "certified": sum(r["status"] == "certified" for r in selected),
            "missing": sum(r["generation_status"] == "missing" for r in selected),
            "unmeasured_or_failed": len(selected)-len(measured), "regressions": sum(r["regression"] for r in selected),
            "coverage": len(measured)/len(selected) if selected else 0,
            "geometric_mean_all_measured_gain": math.exp(statistics.fmean(math.log(g) for g in measured)) if measured else None,
            "gain_stddev_between_campaigns": statistics.stdev(measured) if len(measured)>1 else None,
            "gain_min": min(measured) if measured else None, "gain_max": max(measured) if measured else None,
            "caution": "Includes measured regressions and no-gain results. Missing/invalid campaigns are unknown, never removed from the denominator."}
        aggregates[mode]["per_workload"] = {}
        for workload in plan.get("workloads", WORKLOADS):
            subset = [row for row in selected if row["workload"] == workload]
            values = [row["gain"] for row in subset if row["gain"] is not None and row["gain"] > 0]
            aggregates[mode]["per_workload"][workload] = {"planned": len(subset), "measured": len(values),
                "unmeasured_or_failed": len(subset)-len(values),
                "geometric_mean_gain": math.exp(statistics.fmean(math.log(value) for value in values)) if values else None,
                "gain_stddev_between_independent_seeds": statistics.stdev(values) if len(values)>1 else None,
                "seeds": [{"seed": row["seed"], "gain": row["gain"], "status": row["status"]} for row in subset]}
    expected_groups = {entry.get("group_id") for entry in plan["entries"]}
    complete = all(groups.get(key, {}).get("execution_complete") and groups.get(key, {}).get("all_modes_available") for key in expected_groups)
    direct_comparisons_qualified = True
    method_comparisons = {}
    for key in expected_groups:
        modes = {entry["mode"] for entry in plan["entries"] if entry.get("group_id") == key}
        expected_pairs = {"full_vs_" + mode for mode in modes - {"full"}} if "full" in modes else set()
        if {"pmpp_permanent_only", "no_pmpp"} <= modes:
            expected_pairs.add("pmpp_permanent_only_vs_no_pmpp")
        contrasts = {}
        for name, original in groups.get(key, {}).get("method_comparisons", {}).items():
            contrast = original
            if not isinstance(contrast, dict):
                continue
            contrast = contrast_view(key, name, contrast)
            fields = ("artifact_hash", "candidate_artifact_hash", "incumbent_artifact_hash",
                      "left_artifact_hash", "right_artifact_hash")
            artifacts = [contrast[field] for field in fields if contrast.get(field)]
            # Legacy comparison summaries can omit one or both role hashes.
            # Their predeclared mode labels bind them to the final outcome;
            # complete explicit role hashes remain authoritative in either order.
            if len(artifacts) < 2:
                left, separator, right = name.partition("_vs_")
                if separator:
                    artifacts.extend(row["artifact_hash"] for row in rows
                        if row["group_id"] == key and row["mode"] in (left, right))
            issue = qualification_issue(artifacts)
            if issue["artifact_incident_blocked"]:
                unqualify(contrast, issue)
                contrast["ok"] = False
            contrasts[name] = contrast
        method_comparisons[key] = contrasts
        direct_comparisons_qualified &= all(contrasts.get(name, {}).get("ok") is True for name in expected_pairs)
    artifact_evaluations_qualified = complete and all(
        r["final_qualified"] and r["gain"] is not None for r in rows) and direct_comparisons_qualified and not global_artifact_uncertainty
    method_protocols_qualified = False if any(protocol_incidents.values()) else (
        None if any(row.get("status") == "invalid" for row in protocol_incident_observations) else True)
    return {"schema_version": 2, "rows": rows, "aggregates": aggregates,
            "direct_comparisons_qualified": direct_comparisons_qualified,
            "method_comparisons": method_comparisons,
            "method_comparisons_view_kind": "artifact_aware_method_comparison_summary",
            "method_comparisons_scope": "Lightweight display decisions and intervals only; raw measurements remain in each original group. No new scientific measurement or numerical recomputation is performed by this view.",
            "complete": complete, "scientific_qualified": artifact_evaluations_qualified and method_protocols_qualified is True,
            "artifact_evaluations_qualified": artifact_evaluations_qualified,
            "original_scientific_qualified": artifact_evaluations_qualified,
            "method_protocols_qualified": method_protocols_qualified,
            "method_protocols_qualification_scope": "Additive public incident evidence only; absence is not an exhaustive historical protocol audit.",
            "protocol_incident_observations": protocol_incident_observations,
            "artifact_qualification_incidents": deepcopy(artifact_qualification_incidents),
            "artifact_incident_observations": deepcopy(artifact_incident_observations),
            "artifact_qualification_eligibility_known": not global_artifact_uncertainty,
            "artifact_incident_qualification_scope": "Additive incident denials only. Absence of an incident does not prove absence of races or other CUDA errors. Unresolved identity prevents global scientific qualification; it does not invent a disqualified artifact.",
            "claim": "Only shared final experiments support quality comparisons; implementation or unit tests alone do not establish GPU performance."}
