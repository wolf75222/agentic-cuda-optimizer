#!/usr/bin/env python3
"""Isolated DSL/source-macro/launch-coupling control; CPU plan unless --run.

This fixed, model-free experiment supplements the previous launch-only control.
It is excluded from the quality-study cohorts and certifies no general speedup.
Only --run can submit its own fresh Slurm allocation. CPU tests simulate CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.autotuning import Autotuner
from optimizer_agent.campaign import (
    CampaignConfig, Evaluator, SlurmBackend, SlurmConfig, WorkloadContract,
    WorkloadSuite, device_identity, write_json,
)
from optimizer_agent.evaluation import LaunchRule
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.tuning_constraints import compile_constraints
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent
from scripts.gpu_autotuning_acceptance import ReceiptRunner, candidate_receipt

SOURCE = '''// Static control for DSL/source-macro/launch coupling, not model output.
#define TILE 8
extern "C" __global__ void matmul(const float* a, const float* b, float* c, int n) {
    __shared__ float at[TILE][TILE];
    __shared__ float bt[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * TILE + y, col = blockIdx.x * TILE + x;
    float sum = 0.0f;
    for (int base = 0; base < n; base += TILE) {
        at[y][x] = row < n && base + x < n ? a[row * n + base + x] : 0.0f;
        bt[y][x] = base + y < n && col < n ? b[(base + y) * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += at[y][k] * bt[k][x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = sum;
}
'''
DOMAIN = {"TILE": [8, 12, 16]}
CONSTRAINTS = [
    {"op": "bounds", "parameters": ["TILE"], "values": [8, 16]},
    {"op": "divisible", "parameters": ["TILE"], "values": [8]},
]
RECEIPT = "tuning-constraints-acceptance.json"


def macro_value(candidate):
    matches = re.findall(r"(?m)^#define TILE ([0-9]+)$", candidate.source.read_text())
    if len(matches) != 1:
        raise ValueError("The materialized control must contain exactly one integer TILE macro")
    return int(matches[0])


def hardware_preflight(candidate, properties):
    """Include the static shared arrays, beyond the generic launch preflight."""
    reason = Autotuner._preflight(candidate, candidate.parameters, (), properties)
    shared = 2 * macro_value(candidate) ** 2 * 4
    limit = properties.get("max_shared_memory_per_block_bytes", 48 * 1024)
    return reason or ("static shared arrays exceed hardware limit" if shared > limit else None)


def prepare(directory, run_id, *, runner=None, properties=None, benchmark_series=3, launches=10):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(run_id=run_id, experimental_mode="test", workload="matmul",
        search_sizes=(32, 64), final_sizes=(96,), seed=1729,
        benchmark_series=benchmark_series, launches_per_series=launches,
        confirmation_batches=3, required_sanitizers=(), profile=False,
        research_required=False, web_search_required=False)
    suite = WorkloadSuite("matmul", directory / "inputs", config)
    baseline = suite.baseline(REPO / "examples/naive_matmul.cu")
    cases = suite.cases()
    contract = WorkloadContract(run_id, 1, "matmul", suite.signature,
        "Independent NumPy float64 CPU matrix multiplication with float32 output",
        benchmark_series=benchmark_series, launches_per_series=launches, warmup=config.warmup,
        confirmation_batches=3, allowed_precision="float32",
        metadata_json=json.dumps({"precision": "float32", "target_environment": device_identity(properties or {}),
            "acceptance_scope": "DSL/source_macro with coupled block and tile; fresh winner measurement",
            "scope": "CUDA Events kernel only; transfers and compilation excluded"}, sort_keys=True))
    evaluator = Evaluator(runner, contract, suite.oracle,
        {"short": cases[:2], "search": cases,
         "performance": [case for case in cases if case.metadata["n"] in config.search_sizes]})
    template = {"source": SOURCE, "kernel": "matmul",
        "launch_rule": asdict(LaunchRule(kind="square_2d", block=(8, 8, 1), tile_x=8, tile_y=8)),
        "parameter_bindings": [{"parameter": "TILE", "target": target}
            for target in ("source_macro", "block_x", "block_y", "tile_x", "tile_y")]}
    domain = {key: list(values) for key, values in DOMAIN.items()}
    constraints = json.loads(json.dumps(CONSTRAINTS))
    restrictions = compile_constraints(constraints, domain)
    materializer = Autotuner(evaluator, work_directory=directory / "candidates")
    candidates = [materializer.materialize(template, {"TILE": value}) for value in domain["TILE"]]
    plan = {"schema_version": 1, "kind": "targeted_dsl_source_macro_acceptance", "run_id": run_id,
        "search_sizes": list(config.search_sizes), "correctness_sizes": [case.metadata["n"] for case in cases],
        "parameter_space": domain, "constraints": constraints, "parameter_bindings": template["parameter_bindings"],
        "contract": asdict(contract), "contract_hash": evaluator.contract_hash,
        "baseline": candidate_receipt(baseline), "model_calls": 0, "model_tokens": 0,
        "hardware_verified": properties is not None,
        "preflight": [{"candidate": candidate_receipt(candidate), "macro_value": macro_value(candidate),
            "dsl_admitted": all(restriction(candidate.parameters) for restriction in restrictions),
            "hardware_reason": hardware_preflight(candidate, properties) if properties is not None else None,
            "static_shared_memory_bytes": 2 * macro_value(candidate) ** 2 * 4,
            "launches": {str(case.metadata["n"]): candidate.launch_rule.apply(case.metadata, properties)
                         for case in cases}} for candidate in candidates],
        "scope": "Functional DSL/source-macro/launch coupling proof only; no final holdout or general gain claim",
        "excluded_from_quality_study": True}
    write_json(directory / "plan.json", plan)
    return evaluator, baseline, template, domain, restrictions, plan


def _evidence_checks(result, candidates, recording, plan, properties, job_id, launches):
    for candidate in candidates.values():
        candidate.check_unchanged()
    rows = [asdict(row) for row in result.configurations]
    by_tile = {candidate.parameters["TILE"]: candidate for candidate in candidates.values()}
    calls = {tile: [row for row in recording.calls if row["request"]["source"] == str(candidate.source)]
             for tile, candidate in by_tile.items()}
    expected_sizes = set(plan["correctness_sizes"])
    admitted_rows = [row for row in rows if row["parameters"]["TILE"] in (8, 16)]
    selection_ids = {row["measurement"]["evaluation_id"] for row in rows if row.get("measurement")}
    reports = [row.get("report", {}) for row in recording.calls]
    validated_first = []
    exact_launch = []
    compile_evidence = []
    for tile in (8, 16):
        candidate = by_tile.get(tile)
        requests = calls.get(tile, [])
        first = next((index for index, row in enumerate(requests) if row["request"]["mode"] == "benchmark"), None)
        validated_first.append(first is not None and expected_sizes == {
            row["request"]["arguments"][-1]["value"] for row in requests[:first]
            if row["request"]["mode"] == "run"})
        exact_launch.append(bool(candidate and requests) and all(
            {key: row["request"][key] for key in ("grid", "block", "shared_memory_bytes")} ==
            candidate.launch_rule.apply({"n": row["request"]["arguments"][-1]["value"]}, properties)
            for row in requests))
        compile_evidence.append(bool(requests) and all(
            row.get("report", {}).get("ok") is True
            and isinstance(row["report"].get("compile_ms"), (int, float))
            and math.isfinite(row["report"]["compile_ms"]) and row["report"]["compile_ms"] >= 0
            and row["report"].get("kernel_resources", {}).get("static_shared_memory_bytes") == 2 * tile * tile * 4
            for row in requests))
    checks = {
        "entire_fixed_domain_observed": [row["parameters"] for row in rows] == [{"TILE": tile} for tile in (8, 12, 16)],
        "dsl_filters_hardware_valid_tile_12": bool(by_tile.get(12)) and hardware_preflight(by_tile[12], properties) is None
            and len([row for row in rows if row["parameters"] == {"TILE": 12} and row["status"] == "filtered"
                     and row["reason"] == "parameter restriction rejected configuration"]) == 1,
        "filtered_configuration_never_sent_to_gpu": 12 in calls and not calls[12],
        "both_admitted_configurations_validated_and_measured": len(admitted_rows) == 2 and all(
            row["status"] == "measured" and row["validation"].get("ok") is True
            and row["measurement"].get("ok") is True for row in admitted_rows),
        "independent_full_validation_precedes_each_first_benchmark": all(validated_first),
        "three_distinct_source_bytes": len(by_tile) == 3 and len({candidate._source_hash for candidate in candidates.values()}) == 3,
        "macro_and_launch_coupled_exactly": len(by_tile) == 3 and all(
            macro_value(candidate) == tile and candidate.launch_rule.block == (tile, tile, 1)
            and candidate.launch_rule.tile_x == tile and candidate.launch_rule.tile_y == tile
            for tile, candidate in by_tile.items()) and all(exact_launch),
        "native_compilation_or_content_cache_and_static_shared_evidence": all(compile_evidence),
        "raw_series_preserved": bool(reports) and all(len(row["report"].get("latency_samples_ms", [])) == launches
            for row in recording.calls if row["request"]["mode"] == "benchmark"),
        "native_request_ids_unique": len({row["run_id"] for row in recording.calls}) == len(recording.calls),
        "winner_has_an_additional_full_native_measurement": bool(result.best_candidate) and len([
            row for row in calls.get(result.best_candidate.parameters["TILE"], [])
            if row["request"]["mode"] == "benchmark"]) == 2 * len(plan["search_sizes"]) * plan["contract"]["benchmark_series"],
        "fresh_winner_measurement": bool(result.remeasurement and result.remeasurement.get("ok") is True
            and result.remeasurement["evaluation_id"] not in selection_ids and result.best_candidate
            and result.remeasurement["artifact_hash"] == result.best_candidate.artifact_hash),
        "all_requests_owned_job": bool(reports) and all(str(report.get("slurm_job_id")) == str(job_id) for report in reports),
    }
    return checks


def run_acceptance(directory, run_id, execution_config, *, benchmark_series=3, launches=10):
    directory = Path(directory).resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", RECEIPT, "worker/slurm-session.json", "experiments.sqlite")):
        raise ValueError("Use a fresh experiment directory; reconcile existing evidence/allocation without resubmitting")
    # DSL compilation and materialization precede creation of any GPU worker.
    *_, cpu_plan = prepare(directory, run_id, benchmark_series=benchmark_series, launches=launches)
    write_json(directory / "preallocation-cpu-plan.json", cpu_plan)
    session = "dsl-source-macro-" + uuid.uuid4().hex
    backend = SlurmBackend(SlurmConfig.load(Path(execution_config)), REPO, directory / "worker", session_id=session)
    provenance = controller_provenance(REPO, __file__, execution_config)
    for support in (REPO / "scripts/gpu_autotuning_acceptance.py", REPO / "examples/naive_matmul.cu"):
        provenance["sources"][str(support.relative_to(REPO))] = hashlib.sha256(support.read_bytes()).hexdigest()
    receipt = {"schema_version": 1, "kind": cpu_plan["kind"], "run_id": run_id,
        "status": "incomplete", "ok": False, "session_id": session, "job_id": None,
        "remote_dir": backend.remote_dir, "source_provenance": provenance,
        "model_calls": 0, "model_tokens": 0, "excluded_from_quality_study": True,
        "scope": cpu_plan["scope"], "preallocation_cpu_plan": str(directory / "preallocation-cpu-plan.json")}
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, provenance)
    started, recording, store = time.monotonic(), None, None
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event": "dsl_source_macro_acceptance_started", "job_id": backend.job_id, "session_id": session}), flush=True)
        backend.wait_until_ready()
        raw_runner = backend.runner()
        properties = raw_runner.get_device_properties(0)
        receipt["hardware_identity"], receipt["hardware"] = device_identity(properties), backend.hardware
        recording = ReceiptRunner(raw_runner, directory)
        evaluator, baseline, template, domain, restrictions, plan = prepare(directory, run_id, runner=recording,
            properties=properties, benchmark_series=benchmark_series, launches=launches)
        receipt["contract_hash"], receipt["contract"] = evaluator.contract_hash, asdict(evaluator.contract)
        receipt["constraints"], receipt["parameter_bindings"] = plan["constraints"], template["parameter_bindings"]
        if any(row["hardware_reason"] for row in plan["preflight"]):
            raise RuntimeError("This worker cannot admit the complete hardware-valid acceptance domain")
        store = ExperimentStore(directory / "experiments.sqlite")
        store.initialize_campaign(run_id, {"contract_hash": evaluator.contract_hash,
            "environment": device_identity(properties), "parameter_space": domain, "constraints": plan["constraints"], "kind": receipt["kind"]})
        store.archive_candidate(run_id, candidate_receipt(baseline))
        receipt["baseline_measurement"] = evaluator.benchmark(baseline)
        store.add_evaluation(run_id, receipt["baseline_measurement"])
        write_json(directory / "baseline-measurement.json", receipt["baseline_measurement"])
        if not receipt["baseline_measurement"].get("ok"):
            raise RuntimeError("Fresh naive baseline failed independent validation/measurement")
        candidates = {}
        tuner = Autotuner(evaluator, store, run_id, work_directory=directory / "candidates")
        materialize = tuner.materialize
        def archive(template, parameters):
            candidate = materialize(template, parameters)
            candidates[candidate.candidate_id] = candidate
            store.archive_candidate(run_id, candidate_receipt(candidate))
            return candidate
        tuner.materialize = archive
        # These counts enumerate this fixed control, not a campaign resource cap.
        receipt["local_search_controls"] = {"configurations": 3, "compilations": 2}
        result = tuner.search(template, domain, receipt["local_search_controls"],
            restrictions=restrictions, hardware_limits=properties)
        receipt["configurations"] = [asdict(row) for row in result.configurations]
        receipt["termination_reason"] = result.termination_reason
        receipt["candidates"] = {key: candidate_receipt(value) for key, value in candidates.items()}
        receipt["selection_measurement"], receipt["winner_remeasurement"] = result.evaluation, result.remeasurement
        receipt["winner"] = candidate_receipt(result.best_candidate) if result.best_candidate else None
        for row in receipt["configurations"]:
            for key in ("validation", "measurement"):
                if row.get(key):
                    store.add_evaluation(run_id, row[key])
        if result.remeasurement:
            store.add_evaluation(run_id, result.remeasurement)
        receipt["checks"] = _evidence_checks(result, candidates, recording, plan, properties, backend.job_id, launches)
        receipt["checks"]["no_model_invocations"] = store.budget_state(run_id)["spent"].get("model_calls", 0) == 0
        write_json(directory / "tuning-search.json", receipt)
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if recording:
            receipt["usage"] = recording.usage()
        if store:
            receipt["ledger_usage"] = store.budget_state(run_id)
            write_json(directory / "events.json", store.events(run_id))
        if backend.job_id:
            try:
                backend.close()
                ended = backend.wait_until_finished(timeout=60)
                if not ended.get("terminal"):
                    backend.close(cancel=True)
                    ended = backend.wait_until_finished(timeout=60)
                receipt["terminal"] = ended
            except Exception as error:
                receipt["cleanup_error"] = str(error)
        ended = receipt.get("terminal", {})
        receipt["helper_wall_seconds"] = time.monotonic() - started
        receipt["ok"] = bool(not receipt.get("error") and not receipt.get("cleanup_error")
            and receipt.get("checks") and all(receipt["checks"].values()) and ended.get("terminal") is True
            and ended.get("state") == "COMPLETED" and ended.get("exit_code") == "0:0"
            and str(ended.get("job_id")) == str(receipt["job_id"]))
        receipt["status"] = "dsl_source_macro_acceptance_passed" if receipt["ok"] else "incomplete"
        write_json(directory / RECEIPT, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly submit this isolated GPU control; default is CPU-only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="dsl-source-macro-" + uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    parser.add_argument("--benchmark-series", type=int, default=3)
    parser.add_argument("--launches", type=int, default=10)
    args = parser.parse_args(argv)
    directory = (args.output or REPO / "results/gpu-tuning-constraints-acceptance" / args.run_id).resolve()
    if (directory / "allocation-intent.json").exists():
        raise ValueError("Use a fresh experiment directory; reconcile the recorded allocation intent")
    if args.run:
        result = run_acceptance(directory, args.run_id, args.execution_config,
            benchmark_series=args.benchmark_series, launches=args.launches)
        print(json.dumps({"ok": result["ok"], "status": result["status"], "job_id": result["job_id"],
                          "receipt": str(directory / RECEIPT)}, indent=2))
        return 0 if result["ok"] else 1
    *_, plan = prepare(directory, args.run_id, benchmark_series=args.benchmark_series, launches=args.launches)
    print(json.dumps({"plan": str(directory / "plan.json"), "submitted": 0, "model_calls": 0,
                      "admitted_tiles": [row["macro_value"] for row in plan["preflight"] if row["dsl_admitted"]],
                      "filtered_tiles": [row["macro_value"] for row in plan["preflight"] if not row["dsl_admitted"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
