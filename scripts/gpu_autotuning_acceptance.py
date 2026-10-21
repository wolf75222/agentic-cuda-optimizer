#!/usr/bin/env python3
"""Targeted AC-12 proof: three actual GEMM launch configurations on a fresh GPU.

Without --run this command prepares an inspectable CPU-only plan and submits
zero jobs. Execution uses the existing Slurm worker, independent oracle and
Autotuner; the unique allocation belongs only to this acceptance experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig, Evaluator, SlurmBackend, SlurmConfig, WorkloadContract,
    WorkloadSuite, device_identity, write_json,
)
from optimizer_agent.autotuning import Autotuner
from optimizer_agent.experiment_store import ExperimentStore
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent


def candidate_receipt(candidate):
    return {"candidate_id": candidate.candidate_id, "source_path": str(candidate.source),
            "source": candidate.source.read_text(), "source_hash": candidate._source_hash,
            "artifact_hash": candidate.artifact_hash, "kernel": candidate.kernel,
            "launch_rule": asdict(candidate.launch_rule), "parameters": dict(candidate.parameters)}


def prepare(directory, run_id, *, runner=None, properties=None, benchmark_series=3, launches=10):
    """Build the fixed experiment without contacting SSH, CUDA or a model."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(run_id=run_id, experimental_mode="test", workload="matmul",
        search_sizes=(64, 128, 256), final_sizes=(96,), seed=1729,
        benchmark_series=benchmark_series, launches_per_series=launches,
        confirmation_batches=3, required_sanitizers=(), profile=False,
        research_required=False, web_search_required=False)
    suite = WorkloadSuite("matmul", directory / "inputs", config)
    baseline = suite.baseline(REPO / "examples/naive_matmul.cu")
    cases = suite.cases()
    performance = [case for case in cases if case.metadata["n"] in config.search_sizes]
    contract = WorkloadContract(run_id, 1, "matmul", suite.signature,
        "Independent NumPy float64 CPU matrix multiplication with float32 output",
        benchmark_series=benchmark_series, launches_per_series=launches, warmup=config.warmup,
        confirmation_batches=3, allowed_precision="float32",
        metadata_json=json.dumps({"precision": "float32", "layout": "row-major contiguous square GEMM",
            "scope": "CUDA Events kernel only; reset before each launch; transfers/compilation excluded",
            "target_environment": device_identity(properties or {}),
            "acceptance_scope": "Three launch bindings; fresh winner measurement; functional autotuning proof"}, sort_keys=True))
    evaluator = Evaluator(runner, contract, suite.oracle,
        {"short": cases[:2], "search": cases, "performance": performance})
    template = {"source": baseline.source.read_text(), "kernel": "matmul",
        "launch_rule": {**asdict(baseline.launch_rule), "block": [16, 8, 1]},
        "parameter_bindings": [{"parameter": "BLOCK_X", "target": "block_x"}]}
    domain = {"BLOCK_X": [8, 16, 32]}
    plan = {"schema_version": 1, "run_id": run_id, "kind": "targeted_gpu_autotuning_acceptance",
        "search_sizes": list(config.search_sizes), "correctness_sizes": [case.metadata["n"] for case in cases],
        "parameter_space": domain, "parameter_bindings": template["parameter_bindings"],
        "fixed_block_y": 8, "launch_rule": template["launch_rule"],
        "baseline": candidate_receipt(baseline), "contract": asdict(contract), "contract_hash": evaluator.contract_hash,
        "model_calls": 0, "model_tokens": 0, "certification_scope": "Functional AC-12 acceptance; no optimizer gain certification"}
    write_json(directory / "plan.json", plan)
    return evaluator, baseline, template, domain, plan


class ReceiptRunner:
    def __init__(self, runner, directory):
        self.runner, self.directory, self.calls = runner, Path(directory), []

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def run(self, run_id, request):
        receipt = {"run_id": run_id, "request": asdict(request)}
        try:
            receipt["report"] = self.runner.run(run_id, request)
            return receipt["report"]
        except Exception as error:
            receipt["error"] = str(error)
            raise
        finally:
            self.calls.append(receipt)
            write_json(self.directory / "request-receipts" / (run_id + ".json"), receipt)

    def usage(self):
        reports = [row.get("report", {}) for row in self.calls]
        return {"harness_requests": len(self.calls),
            "benchmark_requests": sum(row["request"]["mode"] == "benchmark" for row in self.calls),
            "native_reservation_wall_seconds": sum(row.get("reservation_wall_seconds", 0) for row in reports),
            "cuda_event_kernel_seconds": sum(sum(row.get("latency_samples_ms", [])) / 1000 for row in reports),
            "native_compile_stage_wall_seconds": sum((row.get("compile_ms") or 0) / 1000 for row in reports),
            "native_cache_hits": sum(row.get("compile_cache_hit") is True for row in reports),
            "native_cache_misses": sum(row.get("compile_cache_hit") is False for row in reports)}


def run_acceptance(directory, run_id, execution_config, *, benchmark_series=3, launches=10, search_timeout=None):
    directory = Path(directory).resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", "autotuning-acceptance.json", "worker/slurm-session.json")):
        raise ValueError("Use a fresh experiment directory; existing GPU evidence/allocation is never overwritten or reused")
    directory.mkdir(parents=True, exist_ok=True)
    # A fresh random session cannot attach to or stop the protected main run.
    session = "ac12-autotune-" + uuid.uuid4().hex
    backend = SlurmBackend(SlurmConfig.load(Path(execution_config)), REPO, directory / "worker", session_id=session)
    receipt = {"schema_version": 1, "kind": "targeted_gpu_autotuning_acceptance", "run_id": run_id,
        "ok": False, "status": "incomplete", "session_id": session, "job_id": None,
        "model_calls": 0, "model_tokens": 0,
        "scope": "Independent CPU oracle; three launch configurations; fresh baseline and selected-winner remeasurement",
        "certification_scope": "Functional AC-12 proof, separate from the optimizer/vanilla comparison and general quality studies"}
    receipt["remote_dir"] = backend.remote_dir
    receipt["source_provenance"] = controller_provenance(REPO, __file__, execution_config)
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    started, recording, store = time.monotonic(), None, None
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event": "autotuning_acceptance_started", "job_id": backend.job_id,
                                      "session_id": session}), flush=True)
        backend.wait_until_ready()
        runner = backend.runner()
        properties = runner.get_device_properties(0)
        receipt["hardware_identity"] = device_identity(properties)
        receipt["hardware"] = backend.hardware
        recording = ReceiptRunner(runner, directory)
        evaluator, baseline, template, domain, plan = prepare(directory, run_id, runner=recording,
            properties=properties, benchmark_series=benchmark_series, launches=launches)
        receipt["contract_hash"], receipt["contract"] = evaluator.contract_hash, asdict(evaluator.contract)
        store = ExperimentStore(directory / "experiments.sqlite")
        store.initialize_campaign(run_id, {"contract_hash": evaluator.contract_hash,
            "environment": device_identity(properties), "parameter_space": domain, "kind": receipt["kind"]})
        store.archive_candidate(run_id, candidate_receipt(baseline))
        measurement = evaluator.benchmark(baseline)
        receipt["baseline_measurement"] = measurement
        write_json(directory / "baseline-measurement.json", measurement)
        store.add_evaluation(run_id, measurement)
        if not measurement.get("ok"):
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
        controls = {"configurations": 3, "compilations": 3}
        if search_timeout is not None:
            controls["wall_seconds"] = search_timeout
        receipt["local_search_controls"] = controls
        result = tuner.search(template, domain, controls, hardware_limits=properties)
        observations = [asdict(row) for row in result.configurations]
        receipt["configurations"], receipt["termination_reason"] = observations, result.termination_reason
        receipt["candidates"] = {key: candidate_receipt(value) for key, value in candidates.items()}
        for row in observations:
            for key in ("validation", "measurement"):
                if row.get(key):
                    store.add_evaluation(run_id, row[key])
        receipt["selection_measurement"], receipt["winner_remeasurement"] = result.evaluation, result.remeasurement
        if result.remeasurement:
            store.add_evaluation(run_id, result.remeasurement)
        receipt["winner"] = candidate_receipt(result.best_candidate) if result.best_candidate else None
        ids = [row["measurement"]["evaluation_id"] for row in observations if row.get("measurement")]
        receipt["checks"] = {
            "all_three_launch_bindings_evaluated": [row["parameters"] for row in observations] == [{"BLOCK_X": x} for x in (8, 16, 32)],
            "every_configuration_validated_before_measurement": len(observations) == 3 and all(
                row["status"] == "measured" and row["validation"].get("ok") and row["measurement"].get("ok") for row in observations),
            "source_unchanged_across_launch_configurations": len(candidates) == 3 and all(
                candidate._source_hash == baseline._source_hash for candidate in candidates.values()),
            "distinct_launches_recorded": sorted(candidate.launch_rule.block for candidate in candidates.values()) == [(8, 8, 1), (16, 8, 1), (32, 8, 1)],
            "fresh_winner_measurement": bool(result.remeasurement and result.remeasurement.get("ok")
                and result.remeasurement["evaluation_id"] not in ids
                and result.best_candidate and result.remeasurement["artifact_hash"] == result.best_candidate.artifact_hash),
            "no_model_invocations": store.budget_state(run_id)["spent"].get("model_calls", 0) == 0,
        }
        write_json(directory / "autotuning-search.json", receipt)
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
        receipt["ok"] = (not receipt.get("error") and not receipt.get("cleanup_error")
            and bool(receipt.get("checks")) and all(receipt["checks"].values()) and ended.get("terminal") is True
            and ended.get("state") == "COMPLETED" and ended.get("exit_code") == "0:0"
            and str(ended.get("job_id")) == str(receipt["job_id"]))
        receipt["status"] = "autotuning_acceptance_passed" if receipt["ok"] else "incomplete"
        write_json(directory / "autotuning-acceptance.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Submit this separate acceptance experiment; default only prepares the CPU plan")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="ac12-autotune-" + uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    parser.add_argument("--benchmark-series", type=int, default=3)
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--search-timeout", type=float, default=None,
                        help="Optional explicit phase timeout in seconds; default has no cumulative search-time limit")
    args = parser.parse_args(argv)
    directory = (args.output or REPO / "results/gpu-autotuning-acceptance" / args.run_id).resolve()
    if (directory / "allocation-intent.json").exists():
        raise ValueError("Use a fresh experiment directory; reconcile the recorded allocation intent")
    if args.run:
        result = run_acceptance(directory, args.run_id, args.execution_config,
            benchmark_series=args.benchmark_series, launches=args.launches, search_timeout=args.search_timeout)
        print(json.dumps({"ok": result["ok"], "status": result["status"], "job_id": result["job_id"],
            "receipt": str(directory / "autotuning-acceptance.json")}, indent=2))
        return 0 if result["ok"] else 1
    prepare(directory, args.run_id, benchmark_series=args.benchmark_series, launches=args.launches)
    print(json.dumps({"plan": str(directory / "plan.json"), "submitted": 0,
                                   "model_calls": 0, "configurations": 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
