"""Explicit six-family, repeated-campaign and isolated-ablation protocol.

Planning never submits a job. Execution is an explicit CLI command; reports
include missing/failed campaigns rather than silently omitting them.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

from .campaign_config import CampaignConfig

WORKLOADS = ("reduction", "scan", "stencil", "convolution", "matmul", "normalization")
MODES = ("full", "greedy_parent", "no_autotuning", "no_research", "no_pmpp")


def plan_suite(base: CampaignConfig, seeds=(1729, 2718, 3141)):
    if len(set(seeds)) < 2:
        raise ValueError("Quality evaluation requires multiple independent campaigns")
    entries = []
    for workload in WORKLOADS:
        for seed in seeds:
            for mode in MODES:
                updates = dict(workload=workload, seed=int(seed),
                    run_id=f"{base.run_id}.{workload}.{seed}.{mode}", profile=False,
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
                config = replace(base, **updates)
                entries.append({"workload": workload, "seed": seed, "mode": mode, "config": config.to_dict()})
    return {"schema_version": 1, "kind": "quality_ablation_plan", "entries": entries,
            "comparison_protocol": "Same iteration/repair/measurement settings; actual GPU and model usage recorded without global resource caps",
            "reference_scope": "greedy_parent recreates original incumbent-parent policy, not the entire unmodified upstream system",
            "upstream_reference": "Run upstream separately, retain its resource receipts, and evaluate exported kernels with the same immutable oracle/protocol; no upstream performance is invented",
            "global_resource_caps": None}


def summarize_suite(plan, manifests):
    rows, aggregates = [], {}
    for entry in plan["entries"]:
        run_id = entry["config"]["run_id"]
        result = manifests.get(run_id)
        row = {k: entry[k] for k in ("workload", "seed", "mode")}
        row.update(run_id=run_id, status="missing", gain=None, model_tokens=None, gpu_seconds=None)
        if result:
            row["status"] = result["certification_status"]
            interval = result.get("final", {}).get("confirmation", {}).get("interval", {})
            row["gain"] = interval.get("gain")
            row["lower"] = interval.get("lower")
            row["upper"] = interval.get("upper")
            if row["status"] == "validated_no_gain":
                row["gain"] = 1.0
            spent = result.get("usage", result.get("budgets", {}).get("spent", {}))
            row.update(model_tokens=spent.get("model_tokens"), gpu_seconds=spent.get("gpu_seconds"),
                       configurations=spent.get("local_configurations", spent.get("configurations", 0)),
                       wall_seconds=result.get("elapsed_seconds"))
        rows.append(row)
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        gains = [r["gain"] for r in selected if r["status"] == "certified" and r["gain"] is not None and r["gain"] > 0]
        aggregates[mode] = {"planned": len(selected), "completed": sum(r["status"] != "missing" for r in selected),
            "certified": len(gains), "missing": sum(r["status"] == "missing" for r in selected),
            "geometric_mean_certified_gain": math.exp(statistics.fmean(math.log(g) for g in gains)) if gains else None,
            "certified_gain_stddev": statistics.stdev(gains) if len(gains)>1 else None,
            "caution": "Certified-only aggregate excludes failures; inspect all rows and missing counts before any quality claim"}
    return {"schema_version": 1, "rows": rows, "aggregates": aggregates,
            "complete": all(r["status"] != "missing" for r in rows),
            "claim": "No general improvement claim from incomplete, uncertified or single-campaign evidence"}
