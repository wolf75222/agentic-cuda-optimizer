#!/usr/bin/env python3
"""Compare four frozen CUDA artifacts on new, post-generation inputs and one GPU.

The default prepares a CPU-only plan. --run creates its own Slurm allocation;
all correctness, sanitizer and paired timing evidence is new. Existing pilot
receipts are retained as source provenance and never supply timing samples.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig, Candidate, Evaluator, LaunchRule, SlurmBackend, SlurmConfig,
    WorkloadContract, WorkloadSuite, device_identity, load_direct_candidates, read_json, write_json,
)
from scripts.gpu_autotuning_acceptance import ReceiptRunner, candidate_receipt

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
ROLES = ("baseline", "agent", "single_prompt", "native_codex")
PAIRS = (("agent_vs_baseline", "agent", "baseline"),
         ("single_prompt_vs_baseline", "single_prompt", "baseline"),
         ("native_codex_vs_baseline", "native_codex", "baseline"),
         ("agent_vs_native_codex", "agent", "native_codex"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def seed_commitment(seed):
    return sha256(str(seed).encode("ascii"))


def select_shapes(seed, search_sizes=(), previous_final_sizes=()):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("The committed seed must be a nonnegative 63-bit integer")
    forbidden = set(search_sizes) | set(previous_final_sizes) | {96, 192, 384, 128, 256, 512}
    rng = np.random.default_rng(seed)
    shapes = []
    for low, high in ((65, 127), (129, 255), (257, 511)):
        if all(n in forbidden for n in range(low, high+1)):
            raise ValueError("No independent shape remains in a prescribed final window")
        n = int(rng.integers(low, high+1))
        while n in forbidden:
            n = int(rng.integers(low, high+1))
        shapes.append(n)
    return tuple(shapes)


def load_config(main_directory, config_path=None):
    if config_path:
        config = CampaignConfig.load(Path(config_path))
    else:
        config = CampaignConfig.load(main_directory / "agent/config.json")
        if not config.run_id.endswith(".agent"):
            raise ValueError("Cannot recover the main run identity from agent/config.json")
        config = replace(config, run_id=config.run_id[:-6])
    if config.workload != "matmul":
        raise ValueError("This independent comparison has an explicit square float32 GEMM contract")
    return config


def native_candidate(path):
    path = Path(path).resolve()
    receipt = read_json(path)
    source_path = receipt.get("source_path", receipt.get("source"))
    if not isinstance(source_path, str):
        raise ValueError("Native candidate requires a frozen source_path")
    source = Path(source_path)
    if not source.is_absolute():
        source = path.parent / source
    candidate = Candidate(receipt["candidate_id"], source, receipt["kernel"],
                          LaunchRule(**receipt["launch_rule"]), receipt["parameters"])
    if candidate._source_hash != receipt.get("source_hash") or candidate.artifact_hash != receipt.get("artifact_hash"):
        raise ValueError("Native source/launch/parameters do not match the frozen candidate receipt")
    if receipt.get("provenance", {}).get("kind") != "native_codex_one_prompt":
        raise ValueError("The native baseline must retain its one-prompt Codex provenance")
    return candidate, receipt


def prepare(main_directory, native_path, seed, output, *, config_path=None, expected_commitment=None):
    main_directory, output = Path(main_directory).resolve(), Path(output).resolve()
    if output == main_directory or output.is_relative_to(main_directory):
        raise ValueError("Use a separate output directory; existing pilot evidence is preserved")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be fresh; completed or partial evidence is never overwritten")
    if expected_commitment is not None and expected_commitment != seed_commitment(seed):
        raise ValueError("The revealed seed differs from the previously published commitment")
    original = load_config(main_directory, config_path)
    # The terminal marker establishes that the main controller has stopped using
    # this allocation. Failed previous pilots remain valid negative provenance.
    terminal = read_json(main_directory / "worker-terminal.json")
    if terminal.get("terminal") is not True:
        raise ValueError("The prior main allocation must be terminal before this final evaluation")
    generated = load_direct_candidates(original, main_directory)
    native, native_receipt = native_candidate(native_path)
    baseline_path = main_directory / "agent/evaluator_data/baseline.cu"
    other_baseline = main_directory / "vanilla/evaluator_data/baseline.cu"
    if not baseline_path.is_file() or not other_baseline.is_file() or baseline_path.read_bytes() != other_baseline.read_bytes():
        raise ValueError("Both original generation modes must identify the same frozen naive baseline source")
    baseline = Candidate("baseline", baseline_path, "matmul", LaunchRule(kind="square_2d", block=(16,16,1)))
    originals = {"baseline": baseline, "agent": generated["agent"],
                 "single_prompt": generated["vanilla"], "native_codex": native}
    if len({candidate.candidate_id for candidate in originals.values()}) != len(ROLES):
        raise ValueError("Each frozen comparison role requires a distinct candidate identity")
    shapes = select_shapes(seed, original.search_sizes, original.final_sizes)
    output.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for role, candidate in originals.items():
        source = output / "sources" / (role + ".cu")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(candidate.source.read_bytes())
        frozen = Candidate(candidate.candidate_id, source, candidate.kernel, candidate.launch_rule, dict(candidate.parameters))
        if frozen.artifact_hash != candidate.artifact_hash:
            raise ValueError("Frozen copy changed an executable artifact")
        candidates[role] = frozen
    final_config = replace(original, run_id="frozen-" + uuid.uuid4().hex[:16], seed=seed,
        final_sizes=shapes, required_sanitizers=TOOLS, experimental_mode="test",
        research_required=False, web_search_required=False, profile=False)
    suite = WorkloadSuite("matmul", output / "inputs", final_config)
    cases = suite.cases(final=True)
    hashes = {str(path.relative_to(output)): sha256(path.read_bytes())
              for path in sorted((output / "inputs").rglob("*.bin"))}
    provenance = {
        "baseline": {"kind": "frozen_naive_reference", "original_source": str(baseline_path)},
        "agent": {"kind": "langgraph_treequest_research_agent", "original_source": str(generated["agent"].source)},
        "single_prompt": {"kind": "generation_only_structured_single_prompt", "original_source": str(generated["vanilla"].source)},
        "native_codex": native_receipt["provenance"],
    }
    previous = {}
    for relative in ("comparison.json", "worker-terminal.json", "agent/config.json", "vanilla/config.json",
                     "agent/manifest.json", "vanilla/manifest.json"):
        path = main_directory / relative
        if path.is_file():
            data = path.read_bytes()
            previous[relative] = {"source_path": str(path), "sha256": sha256(data)}
            destination = output / "provenance" / "generation-only-pilot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    native_bytes = Path(native_path).resolve().read_bytes()
    write_json(output / "provenance/native-candidate.json", native_receipt)
    plan = {"schema_version": 1, "kind": "independent_frozen_kernel_comparison", "phase": "planned",
        "seed": seed, "seed_commitment": seed_commitment(seed), "seed_commitment_encoding": "sha256(decimal ASCII seed)",
        "expected_pre_generation_commitment": expected_commitment,
        "shape_generator": "numpy.default_rng(seed), inclusive windows 65..127/129..255/257..511, reject prior shapes",
        "shapes": list(shapes), "shape_commitment": sha256(json.dumps(shapes, separators=(",", ":")).encode()),
        "excluded_shapes": sorted(set(original.search_sizes) | set(original.final_sizes) | {96,192,384,128,256,512}),
        "input_hashes": hashes, "cases": [{"case_id": case.case_id, "seed": case.seed, "metadata": case.metadata} for case in cases],
        "candidates": {role: {**candidate_receipt(candidate), "provenance": provenance[role]} for role,candidate in candidates.items()},
        "prior_receipts": previous, "native_candidate_receipt": {"path": str(Path(native_path).resolve()), "sha256": sha256(native_bytes)},
        "protocol": {"benchmark_series": final_config.benchmark_series, "confirmation_batches": final_config.confirmation_batches,
            "launches_per_series": final_config.launches_per_series, "warmup": final_config.warmup,
            "confidence": final_config.confidence_level, "min_relative_gain": final_config.min_relative_gain,
            "max_critical_regression": final_config.max_critical_regression, "required_sanitizers": list(TOOLS)},
        "comparisons": [{"name": name, "candidate": candidate, "reference": reference,
                         "gain_direction": f"{reference} latency / {candidate} latency; >1 favors {candidate}"}
                        for name,candidate,reference in PAIRS],
        "model_calls": 0, "timing_source": "All pairs will be measured anew on one fresh physical GPU; previous timing series are never pooled",
        "generation_boundary": "Invoke only after all generators have ended; final cases are never sent to a generator"}
    write_json(output / "plan.json", plan)
    return final_config, suite, cases, candidates, plan


def make_evaluator(config, suite, cases, runner, properties):
    contract = WorkloadContract(config.run_id, 1, "matmul", suite.signature,
        "Independent NumPy float64 CPU matrix multiplication with float32 outputs on new committed-seed inputs",
        benchmark_series=config.benchmark_series, confirmation_batches=config.confirmation_batches,
        launches_per_series=config.launches_per_series, warmup=config.warmup, confidence=config.confidence_level,
        min_relative_gain=config.min_relative_gain, max_critical_regression=config.max_critical_regression,
        required_sanitizers=TOOLS, allowed_precision="float32",
        metadata_json=json.dumps({"precision": "float32", "layout": "row-major contiguous square GEMM",
            "target_environment": device_identity(properties), "input_seed": config.seed,
            "shapes": list(config.final_sizes), "timed_scope": "CUDA Events kernel only; reset every launch",
            "cache_regime": "identical per candidate; input reset and transfers outside event timing",
            "comparison_scope": "Four frozen artifacts, no generation or outcome-dependent retries"}, sort_keys=True))
    return Evaluator(runner, contract, suite.oracle, {"short": cases[:1], "search": cases, "performance": cases})


def execute(prepared, output, execution_config, *, walltime="24:00:00"):
    config, suite, cases, candidates, plan = prepared
    output = Path(output).resolve()
    marker = output / "execution-started.json"
    if marker.exists() or (output / "comparison.json").exists():
        raise ValueError("This frozen comparison already started; preserve its outcome instead of repeating for a win")
    session = "frozen-finals-" + uuid.uuid4().hex
    write_json(marker, {"session_id": session, "seed_commitment": plan["seed_commitment"], "started_at_unix": time.time()})
    backend = SlurmBackend(replace(SlurmConfig.load(Path(execution_config)), walltime=walltime),
                           REPO, output / "worker", session_id=session)
    proof = {**plan, "phase": "running", "ok": False, "status": "incomplete", "session_id": session,
             "job_id": None, "validations": {}, "sanitizer_qualification": {}, "paired_comparisons": {}}
    started, recording = time.monotonic(), None
    try:
        proof["job_id"] = backend.start()
        print(json.dumps({"event": "frozen_comparison_started", "job_id": backend.job_id, "session_id": session}), flush=True)
        backend.wait_until_ready()
        runner = backend.runner()
        properties = runner.get_device_properties(0)
        proof["hardware_identity"], proof["hardware"] = device_identity(properties), backend.hardware
        recording = ReceiptRunner(runner, output)
        evaluator = make_evaluator(config, suite, cases, recording, properties)
        proof["contract"], proof["contract_hash"] = asdict(evaluator.contract), evaluator.contract_hash
        write_json(output / "comparison.json", proof)  # Hardware and contract frozen before the first launch.
        for role in ROLES:
            short = evaluator.validate(candidates[role], "short")
            complete = evaluator.validate(candidates[role], "search") if short.get("ok") else short
            proof["validations"][role] = {"short": short, "complete": complete}
            write_json(output / "comparison.json", proof)
            if not complete.get("ok"):
                raise RuntimeError(f"Frozen candidate failed independent correctness: {role}")
        for role in ROLES:
            qualification = evaluator.qualify_sanitizers(candidates[role], "performance")
            proof["sanitizer_qualification"][role] = qualification
            write_json(output / "comparison.json", proof)
            if qualification.get("ok") is not True or qualification.get("artifact_hash") != candidates[role].artifact_hash:
                raise RuntimeError(f"Required instrumented qualification failed: {role}")
        for name, candidate, reference in PAIRS:
            result = evaluator.confirm(candidates[candidate], candidates[reference])
            result.update(candidate_role=candidate, reference_role=reference,
                gain_direction=f"{reference} latency / {candidate} latency; >1 favors {candidate}",
                interpretation="Fixed fresh paired comparison; decision does not modify any previous campaign incumbent")
            proof["paired_comparisons"][name] = result
            write_json(output / "comparison.json", proof)
            if result.get("ok") is not True or result.get("status") not in {"promoted", "not_improved", "inconclusive"}:
                raise RuntimeError(f"Paired comparison incomplete: {name}")
        current_hashes = {str(path.relative_to(output)): sha256(path.read_bytes())
                          for path in sorted((output / "inputs").rglob("*.bin"))}
        if current_hashes != plan["input_hashes"]:
            raise RuntimeError("Frozen final input bytes changed during evaluation")
        for candidate in candidates.values():
            candidate.check_unchanged()
    except Exception as error:
        proof.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if recording:
            proof["measurement_usage"] = recording.usage()
        instrumented = [row for receipt in proof["sanitizer_qualification"].values()
                        for row in receipt.get("sanitizers", [])]
        proof["sanitizer_usage"] = {"tool_calls": len(instrumented),
            "native_reservation_wall_seconds": sum(row.get("reservation_wall_seconds", 0) for row in instrumented),
            "instrumented_times_excluded_from_comparisons": True}
        if backend.job_id:
            try:
                backend.close()
                terminal = backend.wait_until_finished(timeout=60)
                if not terminal.get("terminal"):
                    backend.close(cancel=True)
                    terminal = backend.wait_until_finished(timeout=60)
                proof["terminal"] = terminal
            except Exception as error:
                proof["cleanup_error"] = str(error)
        terminal = proof.get("terminal", {})
        proof["elapsed_seconds"] = time.monotonic()-started
        proof["ok"] = (not proof.get("error") and not proof.get("cleanup_error")
            and set(proof["sanitizer_qualification"]) == set(ROLES)
            and all(row.get("ok") is True for row in proof["sanitizer_qualification"].values())
            and set(proof["paired_comparisons"]) == {row[0] for row in PAIRS}
            and all(row.get("ok") is True for row in proof["paired_comparisons"].values())
            and terminal.get("terminal") is True and terminal.get("state") == "COMPLETED"
            and terminal.get("exit_code") == "0:0" and str(terminal.get("job_id")) == str(proof["job_id"]))
        proof["phase"] = "finished"
        proof["status"] = "frozen_comparison_complete" if proof["ok"] else "incomplete"
        write_json(output / "comparison.json", proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_campaign", type=Path)
    parser.add_argument("--native-candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Previously committed private seed, revealed only after all generation ends")
    parser.add_argument("--seed-commitment", help="Expected previously published sha256(decimal ASCII seed)")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory outside the previous pilot")
    parser.add_argument("--config", type=Path, help="Original main campaign configuration; default derives agent/config.json")
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    parser.add_argument("--run", action="store_true", help="Start a fresh owned Slurm measurement context; default only builds the CPU plan")
    args = parser.parse_args(argv)
    prepared = prepare(args.main_campaign, args.native_candidate, args.seed, args.output,
                       config_path=args.config, expected_commitment=args.seed_commitment)
    if not args.run:
        print(json.dumps({"plan": str(args.output.resolve()/"plan.json"), "submitted": 0,
                          "seed_commitment": prepared[-1]["seed_commitment"], "shapes": prepared[-1]["shapes"]}, indent=2))
        return 0
    result = execute(prepared, args.output, args.execution_config)
    print(json.dumps({"ok": result["ok"], "status": result["status"], "job_id": result["job_id"],
                      "receipt": str(args.output.resolve()/"comparison.json")}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
