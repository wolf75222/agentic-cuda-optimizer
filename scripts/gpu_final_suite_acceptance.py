#!/usr/bin/env python3
"""Verify standalone finalize() on distinct SEARCH/FINAL cases, without a model.

Default: prepare a CPU-only plan. --run owns one new Slurm allocation. This
declared functional control never contributes samples to the quality study.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig, Candidate, Evaluator, LaunchRule, SlurmBackend, SlurmConfig,
    WorkloadContract, WorkloadSuite, device_identity, write_json,
)
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent
from scripts.gpu_autotuning_acceptance import ReceiptRunner, candidate_receipt

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")


def source_provenance(execution_config):
    value = controller_provenance(REPO, __file__, execution_config)
    for name in ("scripts/gpu_autotuning_acceptance.py", "examples/naive_matmul.cu"):
        value["sources"][name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return value


class FinalReceiptRunner(ReceiptRunner):
    def __init__(self, runner, directory):
        super().__init__(runner, directory)
        self.sanitizers = []

    def run_sanitizer(self, run_id, request, tool):
        row = {"run_id": run_id, "request": asdict(request), "tool": tool}
        try:
            row["report"] = self.runner.run_sanitizer(run_id, request, tool)
            return row["report"]
        except Exception as error:
            row["error"] = str(error)
            raise
        finally:
            self.sanitizers.append(row)
            write_json(self.directory / "sanitizer-receipts" / (run_id + ".json"), row)


def prepare(directory, run_id, *, runner=None, properties=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(run_id=run_id, workload="matmul", experimental_mode="test",
        research_required=False, web_search_required=False, search_sizes=(17, 63),
        final_sizes=(96, 128), benchmark_series=2, confirmation_batches=4,
        launches_per_series=5, warmup=1, required_sanitizers=TOOLS, profile=False)
    suite = WorkloadSuite("matmul", directory / "inputs", config)
    search = suite.cases()
    baseline = suite.baseline(REPO / "examples/naive_matmul.cu")
    candidate = Candidate("declared-final-suite-control", baseline.source, "matmul",
        LaunchRule(kind="square_2d", block=(8, 8, 1)))
    identity = hashlib.sha256(json.dumps({"shapes": [96, 128], "generator": "WorkloadSuite v1",
        "scope": "Declared functional control, not quality holdout"}, sort_keys=True).encode()).hexdigest()
    contract = WorkloadContract(run_id, 1, "matmul", suite.signature,
        "Independent NumPy float64 matrix multiplication with float32 outputs",
        benchmark_series=2, launches_per_series=5, warmup=1, confirmation_batches=4,
        required_sanitizers=TOOLS, allowed_precision="float32",
        metadata_json=json.dumps({"target_environment": device_identity(properties or {}),
            "final_suite_identity": identity, "precision": "float32",
            "scope": "Functional final-suite control; no optimizer quality or general gain claim"}, sort_keys=True))
    evaluator = Evaluator(runner, contract, suite.oracle,
        {"short": search[:1], "search": search, "performance": search},
        final_suite_factory=lambda: suite.cases(final=True), final_suite_identity=identity)
    plan = {"schema_version": 1, "kind": "standalone_distinct_final_suite_acceptance",
        "run_id": run_id, "submitted": False, "model_calls": 0, "model_tokens": 0,
        "scope": "Declared static functional control, excluded from quality cohorts",
        "search_shapes": [17, 63], "final_shapes": [96, 128],
        "required_sanitizers": list(TOOLS), "confirmation_batches": 4,
        "launches_per_series": 5, "contract_hash": evaluator.contract_hash,
        "candidate": candidate_receipt(candidate), "baseline": candidate_receipt(baseline),
        "expected": "All standalone final timings and sanitizer checks use FINAL, never SEARCH cases"}
    write_json(directory / "plan.json", plan)
    return evaluator, baseline, candidate, search, plan


def check_final(evaluator, baseline, candidate, recording):
    """Use the public finalization path; inspect real requests rather than labels."""
    search = evaluator.validate(candidate, "search")
    before = len(recording.calls)
    result = evaluator.finalize(candidate, baseline)
    calls = recording.calls[before:]
    benchmarks = [row for row in calls if row["request"]["mode"] == "benchmark"]
    confirmation = result.get("confirmation", {})
    final_ids = {row["case_id"] for row in result.get("validation", {}).get("cases", [])}
    search_ids = {row["case_id"] for row in search.get("cases", [])}
    pairs = confirmation.get("pairs", [])
    # Requests carry scalar n and the exact original input file paths. A correct
    # receipt label alone cannot hide timing of the old search input buffers.
    shapes = {row["request"]["arguments"][-1].get("value") for row in benchmarks}
    validation_inputs = {tuple(arg.get("input_file") for arg in row["request"]["arguments"][:2])
        for row in calls if row["request"]["mode"] == "run"}
    benchmark_inputs = {tuple(arg.get("input_file") for arg in row["request"]["arguments"][:2]) for row in benchmarks}
    sanitizer_inputs = {tuple(arg.get("input_file") for arg in row["request"]["arguments"][:2]) for row in recording.sanitizers}
    expected_pairs = {(case, batch) for case in final_ids for batch in range(4)}
    before_repeat = (len(recording.calls), len(recording.sanitizers))
    repeat = evaluator.finalize(candidate, baseline)
    checks = {
        "search_oracle_passed": search.get("ok") is True,
        "distinct_artifacts": candidate.artifact_hash != baseline.artifact_hash,
        "final_oracles_passed": result.get("validation", {}).get("ok") is True
            and result.get("baseline_validation", {}).get("ok") is True,
        "two_disjoint_final_cases": len(final_ids) == 2 and not final_ids & search_ids,
        "confirmation_explicitly_final": confirmation.get("suite") == "final" and set(confirmation.get("case_ids", [])) == final_ids,
        "fresh_final_shapes_only": shapes == {96, 128} and len(benchmarks) == 16,
        "final_input_buffers_used": len(benchmark_inputs) == 2 and benchmark_inputs <= validation_inputs,
        "exact_fixed_pairs": len(pairs) == 8 and {(row["case_id"], row["series"]) for row in pairs} == expected_pairs,
        "alternating_AB_BA": all(row["order"] == ([baseline.candidate_id, candidate.candidate_id]
            if row["series"] % 2 == 0 else [candidate.candidate_id, baseline.candidate_id]) for row in pairs),
        "all_four_sanitizers_on_final_cases": len(result.get("sanitizers", [])) == 8
            and {(row.get("tool"), row.get("case_id")) for row in result.get("sanitizers", [])} == {(tool, case) for tool in TOOLS for case in final_ids}
            and all(row.get("ok") is True for row in result.get("sanitizers", [])),
        "sanitizer_requests_use_exact_final_inputs": len(recording.sanitizers) == 8
            and {(row["tool"], row["request"]["arguments"][-1].get("value")) for row in recording.sanitizers} == {(tool, n) for tool in TOOLS for n in (96, 128)}
            and len(sanitizer_inputs) == 2 and sanitizer_inputs <= validation_inputs,
        "finalization_completed": result.get("ok") is True and result.get("status") in {"certified", "validated_gain_uncertified"},
        "exposed_holdout_not_reused": repeat.get("ok") is False and repeat.get("status") == "finalization_incomplete"
            and before_repeat == (len(recording.calls), len(recording.sanitizers)),
        "final_authorization_revoked": evaluator._final_timing_authorized is False,
    }
    return {"ok": all(checks.values()), "checks": checks, "search_validation": search,
        "finalization": result, "repeat_refusal": repeat, "final_harness_requests": len(calls),
        "final_benchmark_requests": len(benchmarks), "instrumented_timings_excluded": True}


def run_acceptance(directory, run_id, execution_config):
    directory = Path(directory).resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", "final-suite-acceptance.json", "worker/slurm-session.json")):
        raise ValueError("Existing allocation/evidence: reconcile its recorded session instead of resubmitting")
    backend = SlurmBackend(SlurmConfig.load(execution_config), REPO, directory / "worker",
        session_id="ac-final-suite-" + uuid.uuid4().hex)
    receipt = {"schema_version": 1, "kind": "standalone_distinct_final_suite_acceptance",
        "run_id": run_id, "ok": False, "status": "incomplete", "job_id": None,
        "session_id": backend.session_id, "remote_dir": backend.remote_dir,
        "model_calls": 0, "model_tokens": 0, "scope": "Functional regression control; excluded from quality cohorts"}
    receipt["source_provenance"] = source_provenance(execution_config)
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    recording, started = None, time.monotonic()
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event": "final_suite_acceptance_started", "job_id": backend.job_id, "session_id": backend.session_id}), flush=True)
        backend.wait_until_ready()
        recording = FinalReceiptRunner(backend.runner(), directory)
        receipt["hardware"] = backend.hardware
        evaluator, baseline, candidate, _, plan = prepare(directory / "controls", run_id,
            runner=recording, properties=recording.get_device_properties(0))
        receipt["plan"] = plan
        receipt["result"] = check_final(evaluator, baseline, candidate, recording)
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if recording:
            receipt["usage"] = recording.usage()
            receipt["usage"]["sanitizer_requests"] = len(recording.sanitizers)
        if backend.job_id:
            try:
                backend.close()
                receipt["worker_terminal"] = backend.wait_until_finished(timeout=60)
            except Exception as error:
                receipt["cleanup_error"] = str(error)
        after = source_provenance(execution_config)
        receipt["sources_unchanged"] = after["sources"] == receipt["source_provenance"]["sources"]
        receipt["source_provenance_after"] = after
        terminal = receipt.get("worker_terminal", {})
        receipt["helper_wall_seconds"] = time.monotonic() - started
        receipt["ok"] = bool(receipt.get("result", {}).get("ok") and receipt["sources_unchanged"]
            and not receipt.get("error") and not receipt.get("cleanup_error")
            and terminal.get("terminal") is True and terminal.get("state") == "COMPLETED"
            and terminal.get("exit_code") == "0:0" and str(terminal.get("job_id")) == str(backend.job_id))
        receipt["status"] = "final_suite_acceptance_passed" if receipt["ok"] else "incomplete"
        write_json(directory / "final-suite-acceptance.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-id", default="final-suite-" + uuid.uuid4().hex[:12])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    args = parser.parse_args(argv)
    if args.run:
        value = run_acceptance(args.output, args.run_id, args.execution_config)
        print(json.dumps({"ok": value["ok"], "status": value["status"], "job_id": value["job_id"]}), flush=True)
        return 0 if value["ok"] else 1
    _, _, _, _, plan = prepare(args.output, args.run_id)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
