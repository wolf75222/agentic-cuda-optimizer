#!/usr/bin/env python3
"""Add actual sanitizer qualification to exact, already measured CUDA artifacts.

This command allocates a new ROMEO worker after the measurement job has ended.
It runs fresh correctness and required sanitizers, never a model or benchmark.
The original measurement contract, timing pairs and confidence interval remain
unchanged; a separate qualification contract and job receipt are attached.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig, SlurmBackend, SlurmConfig, comparison_completed, device_identity,
    digest, direct_evaluator, load_direct_candidates, read_json, write_json,
)


MODES = ("agent", "vanilla")


def input_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((directory / "direct-comparison/inputs").rglob("*.bin")) if path.is_file()}


def measurement_identity(direct):
    return digest({key: value for key, value in direct.items()
                   if key not in ("sanitizer_qualification", "required_sanitizers")})


def qualify_comparison(config, directory, worker_session, *, controller_note=None):
    directory = Path(directory).resolve()
    if config.backend != "romeo" or not config.required_sanitizers:
        raise ValueError("Qualification requires ROMEO and a nonempty configured sanitizer tool set")
    summary_path = directory / "comparison.json"
    direct_path = directory / "direct-comparison/comparison.json"
    proof_path = directory / "direct-comparison/sanitizer-qualification.json"
    summary, direct = read_json(summary_path), read_json(direct_path)
    if measurement_identity(summary.get("generated_agent_vs_vanilla", {})) != measurement_identity(direct):
        raise ValueError("Root and direct measurement receipts differ")
    if direct.get("ok") is not True or not direct.get("pairs") or not direct.get("contract_hash"):
        raise ValueError("A complete original paired measurement receipt is required")
    main_job = str(summary.get("job_id", ""))
    terminal = read_json(directory / "worker-terminal.json")
    if (not main_job or str(terminal.get("job_id")) != main_job or terminal.get("terminal") is not True
            or terminal.get("state") != "COMPLETED" or terminal.get("exit_code") != "0:0"):
        raise ValueError("The exact main measurement job must have successful terminal accounting before qualification")
    for mode in MODES:
        if not (directory / "direct-comparison" / (mode + ".cu")).is_file():
            raise ValueError(f"Missing immutable measured source: {mode}")
    candidates = load_direct_candidates(config, directory)
    artifacts = {mode: candidates[mode].artifact_hash for mode in MODES}
    if artifacts != direct.get("candidate_artifacts"):
        raise ValueError("Generated source/launch/parameter artifacts do not match the paired measurement")
    main_hardware = device_identity(summary.get("hardware", {}).get("device_properties", {}))
    if not main_hardware.get("uuid") or device_identity(direct.get("environment", {})) != main_hardware:
        raise ValueError("The main receipt lacks matching physical GPU/toolchain identity")
    if proof_path.exists() and comparison_completed(summary):
        existing = read_json(proof_path)
        if (existing == direct.get("sanitizer_qualification") and existing.get("ok") is True
                and existing.get("candidate_artifacts") == artifacts
                and str(existing.get("main_measurement_job_id")) == main_job):
            return existing  # Completed evidence is reused without more GPU work.

    before = input_hashes(directory)
    if not before:
        raise ValueError("Original independent direct-comparison input files are missing")
    # Construction only rehydrates deterministic CPU inputs; it runs no GPU.
    # Save the old hash map before rehydration and reject any byte change.
    evaluator = direct_evaluator(config, directory, None)
    after = input_hashes(directory)
    if after != before:
        raise ValueError("Direct-comparison inputs changed while recreating the fixed independent suite")
    paired_cases = {pair["case_id"] for pair in direct["pairs"]}
    cases = evaluator._suites["performance"]
    if {case.case_id for case in cases} != paired_cases:
        raise ValueError("Qualification shapes differ from the original paired measurement")
    measured_protocol = dict(direct.get("protocol", {}))
    qualifier_protocol = asdict(evaluator.contract)
    measured_without_tools = {key: value for key, value in measured_protocol.items() if key != "required_sanitizers"}
    qualifier_without_tools = {key: value for key, value in qualifier_protocol.items() if key != "required_sanitizers"}
    if not measured_protocol or digest(measured_without_tools) != digest(qualifier_without_tools):
        raise ValueError("Qualification changes more than the original contract's required sanitizer tools")

    proof = {
        "schema_version": 1, "ok": False, "status": "sanitizer_qualification_incomplete",
        "scope": "Posthoc correctness and instrumented tools only; original paired timings and intervals are unchanged",
        "proof_reference": "direct-comparison/sanitizer-qualification.json",
        "main_measurement_job_id": main_job, "qualification_job_id": None,
        "candidate_artifacts": artifacts, "required_sanitizers": list(config.required_sanitizers),
        "measurement_contract_hash": direct["contract_hash"],
        "qualification_contract_hash": evaluator.contract_hash,
        "measurement_protocol": measured_protocol, "qualification_protocol": qualifier_protocol,
        "independent_seed_base": config.seed + 3000017,
        "input_hashes": before,
        "cases": [{"case_id": case.case_id, "seed": case.seed, "metadata": case.metadata} for case in cases],
        "main_hardware_identity": main_hardware, "hardware_identity": None,
        "controller_provenance": controller_note or (
            "The completed measurement controller's receipts are retained. Current-release source admission, "
            "including the precision guard, is applied separately by fresh correctness validation in this "
            "posthoc qualification; this does not retroactively change the measurement controller or its contract."),
        "candidates": {}, "ranking_eligible": False, "instrumented_timings_excluded": True,
    }
    backend = SlurmBackend(SlurmConfig.load(REPO / config.execution_config), REPO,
                           directory / "direct-comparison/qualification-worker", session_id=worker_session)
    try:
        proof["qualification_job_id"] = backend.start()
        if str(proof["qualification_job_id"]) == main_job:
            raise ValueError("Qualification requires a fresh job distinct from the finished measurement job")
        print(json.dumps({"event": "qualification_job_started", "job_id": backend.job_id}), flush=True)
        backend.wait_until_ready()
        runner = backend.runner()
        proof["hardware_identity"] = device_identity(runner.get_device_properties(0))
        if proof["hardware_identity"] != main_hardware:
            raise ValueError("Qualification GPU UUID/toolchain differs from the measured artifacts' environment")
        evaluator.runner = runner
        for mode in MODES:
            report = evaluator.qualify_sanitizers(candidates[mode], "performance")
            proof["candidates"][mode] = report
            write_json(proof_path, proof)
            print(json.dumps({"event": "candidate_qualified", "mode": mode,
                              "ok": report.get("ok"), "status": report.get("status")}), flush=True)
            if report.get("ok") is not True or report.get("artifact_hash") != artifacts[mode]:
                raise ValueError(f"Exact candidate sanitizer qualification failed: {mode}")
        if input_hashes(directory) != before:
            raise ValueError("Frozen direct-comparison inputs changed during qualification")
    except Exception as error:
        proof.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if backend.job_id:
            try:
                backend.close()
                proof["terminal_receipt"] = backend.wait_until_finished(timeout=60)
                if not proof["terminal_receipt"].get("terminal"):
                    backend.close(cancel=True)
                    proof["terminal_receipt"] = backend.wait_until_finished(timeout=60)
            except Exception as error:
                proof["cleanup_error"] = str(error)
        ended = proof.get("terminal_receipt", {})
        proof["ok"] = (not proof.get("error") and not proof.get("cleanup_error")
                       and set(proof["candidates"]) == set(MODES)
                       and all(row.get("ok") is True for row in proof["candidates"].values())
                       and ended.get("terminal") is True and ended.get("state") == "COMPLETED"
                       and ended.get("exit_code") == "0:0"
                       and str(ended.get("job_id")) == str(proof["qualification_job_id"]))
        proof["status"] = "sanitizer_qualified" if proof["ok"] else "sanitizer_qualification_incomplete"
        write_json(proof_path, proof)

    # Refuse to attach qualification to measurements changed by another writer.
    current_summary, current_direct = read_json(summary_path), read_json(direct_path)
    if (str(current_summary.get("job_id")) != main_job
            or measurement_identity(current_direct) != measurement_identity(direct)
            or measurement_identity(current_summary.get("generated_agent_vs_vanilla", {})) != measurement_identity(direct)):
        raise ValueError("Measurement receipts changed during qualification; proof saved separately without attachment")
    current_direct["required_sanitizers"] = list(config.required_sanitizers)
    current_direct["sanitizer_qualification"] = proof
    current_summary["generated_agent_vs_vanilla"] = current_direct
    current_summary["ok"] = comparison_completed(current_summary)
    current_summary["status"] = "completed" if current_summary["ok"] else "incomplete"
    write_json(direct_path, current_direct)
    write_json(summary_path, current_summary)
    if proof["ok"] is not True:
        raise RuntimeError("Qualification failed; inspect " + str(proof_path))
    if current_summary["ok"] is not True:
        raise RuntimeError("Sanitizer proof saved, but the complete comparison predicate remains incomplete")
    return read_json(proof_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, help="Defaults to results_dir/run_id from configuration")
    parser.add_argument("--worker-session", required=True, help="New isolated Slurm identity; never the main measurement session")
    parser.add_argument("--controller-note", help="Explicit provenance note for the already executed measurement controller")
    args = parser.parse_args(argv)
    try:
        config = CampaignConfig.load(args.config)
        directory = args.campaign or REPO / config.results_dir / config.run_id
        proof = qualify_comparison(config, directory, args.worker_session, controller_note=args.controller_note)
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": proof["ok"], "proof": str(Path(directory).resolve() / proof["proof_reference"]),
                      "main_measurement_job_id": proof["main_measurement_job_id"],
                      "qualification_job_id": proof["qualification_job_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
