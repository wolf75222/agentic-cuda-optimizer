"""Predeclared §16 design and inclusive, evidence-based reporting.

Generation scores are never substituted for the shared final GPU experiment.
Unknown outcomes remain unknown in aggregates, with their full denominator.
"""
from dataclasses import replace
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


def summarize_suite(plan, manifests, groups=None):
    groups = groups or {}
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
            final_terminal=group.get("worker_terminal"), final_environment=group.get("environment"))
        if result.get("status") == "validated_no_gain":
            row["gain"] = 1.0
            row["gain_basis"] = "exact_baseline_artifact"
        row["regression"] = row["gain"] is not None and row["gain"] < 1
        row["final_qualified"] = bool(result.get("qualified"))
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
    for key in expected_groups:
        modes = {entry["mode"] for entry in plan["entries"] if entry.get("group_id") == key}
        expected_pairs = {"full_vs_" + mode for mode in modes - {"full"}} if "full" in modes else set()
        if {"pmpp_permanent_only", "no_pmpp"} <= modes:
            expected_pairs.add("pmpp_permanent_only_vs_no_pmpp")
        contrasts = groups.get(key, {}).get("method_comparisons", {})
        direct_comparisons_qualified &= all(contrasts.get(name, {}).get("ok") is True for name in expected_pairs)
    return {"schema_version": 2, "rows": rows, "aggregates": aggregates,
            "direct_comparisons_qualified": direct_comparisons_qualified,
            "complete": complete, "scientific_qualified": complete and all(
                r["final_qualified"] and r["gain"] is not None for r in rows) and direct_comparisons_qualified,
            "claim": "Only shared final experiments support quality comparisons; implementation or unit tests alone do not establish GPU performance."}
