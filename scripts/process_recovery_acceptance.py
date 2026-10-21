#!/usr/bin/env python3
"""Crash real processes at result/tell boundaries using archived CUDA evidence.

This replays existing measurements in isolated functional-test ledgers. It
creates no model call or GPU request and makes no new performance observation.
The original campaign, checkpoints and resource accounting remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from optimizer_agent.campaign import read_json, write_json
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.search_policy import SearchPolicy


def policy(directory, fixture):
    return SearchPolicy(ExperimentStore(directory / "recovery.sqlite"), "functional-replay",
        fixture["baseline"], fixture["baseline_latencies"], actions=["tiling"],
        contract_fingerprint=fixture["contract_hash"],
        environment_fingerprint=fixture["environment_hash"], seed=1729)


def child(directory, phase):
    fixture = read_json(directory / "fixture.json")
    current = policy(directory, fixture)
    evidence = fixture["result"]
    if phase in ("after_result", "after_tell"):
        first, pending = current.ask_batch(2)
        write_json(directory / "ids.json", {"first": first.trial_id, "pending": pending.trial_id})
        if phase == "after_result":
            current.integrate_pending = lambda **kwargs: os._exit(87)
        else:
            current._before_tree_commit = lambda *args: os._exit(87)
        current.tell_once(first.trial_id, evidence)
        raise RuntimeError("Crash hook was not reached")
    ids = read_json(directory / "ids.json")
    pending = [trial.trial_id for trial in current.pending_trials()]
    if pending != [ids["pending"]]:
        raise AssertionError("Pending trial was not restored exactly")
    if current.tell_once(ids["first"], evidence) is not False:
        raise AssertionError("Replayed result was integrated twice")
    duplicate = current.ask()
    if current.tell_once(duplicate.trial_id, evidence) is not False:
        raise AssertionError("Two trials claimed the same CUDA measurement")
    current.suspend_or_cancel(ids["pending"], "Functional replay completed; no physical work submitted",
                              consumption={"trials": 1})
    events = current.store.events("functional-replay")
    with current.store.connection() as db:
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("evaluations", "reward_claims", "trials")}
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    receipt = {"tree_nodes": len(current.snapshot()["tree"].tree),
               "tree_reward_events": sum(row["kind"] == "tree_reward" for row in events),
               "duplicate_events": sum(row["kind"] == "duplicate_measurement" for row in events),
               "restored_pending": pending, "counts": counts, "integrity": integrity,
               "usage": current.store.budget_state("functional-replay")["spent"],
               "model_calls": 0, "gpu_requests": 0}
    if (receipt["tree_nodes"] != 2 or receipt["tree_reward_events"] != 1 or
            receipt["duplicate_events"] != 1 or counts["evaluations"] != 1 or
            counts["reward_claims"] != 1 or integrity != "ok"):
        raise AssertionError(receipt)
    write_json(directory / "resumed.json", receipt)


def run_acceptance(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh functional acceptance directory; do not overwrite evidence")
    baseline_path = campaign / "baseline-measurement.json"
    baseline_measurement = read_json(baseline_path)
    result_paths = sorted(campaign.glob("trials/*/result.json"))
    result_path = next((p for p in result_paths if read_json(p).get("tell", {}).get("valid") is True), None)
    if result_path is None or not baseline_measurement.get("ok"):
        raise ValueError("Valid archived independent CUDA baseline and tree-reward result are required")
    result = read_json(result_path)["tell"]
    if result.get("purpose") != "tree_reward" or result.get("fresh") is not True:
        raise ValueError("Source result is not an independently measured tree reward")
    # Replay accounting counts logical trials only. It must never charge the
    # archived allocation/model consumption as new physical work.
    result["consumption"] = {"trials": 1}
    baseline_source = campaign / "evaluator_data/baseline.cu"
    fixture = {"baseline": {"candidate_id": "baseline", "source": baseline_source.read_text(),
                             "parameters": {}, "valid": True},
               "baseline_latencies": {name: row["latency_ms"] for name, row in baseline_measurement["cases"].items()},
               "contract_hash": baseline_measurement["contract_hash"],
               "environment_hash": hashlib.sha256(json.dumps(read_json(campaign / "environment.json"), sort_keys=True).encode()).hexdigest(),
               "result": result}
    source_files = [baseline_path, result_path, campaign / "config.json", campaign / "environment.json", baseline_source]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    output.mkdir(parents=True)
    cases = {}
    for boundary in ("after_result", "after_tell"):
        directory = output / boundary
        directory.mkdir()
        write_json(directory / "fixture.json", fixture)
        command = [sys.executable, str(Path(__file__).resolve()), "--_directory", str(directory)]
        crash = subprocess.run(command + ["--_phase", boundary], capture_output=True, text=True, timeout=60)
        if crash.returncode != 87:
            raise RuntimeError(f"Expected process exit87 at {boundary}: {crash.returncode}: {crash.stderr[-2000:]}")
        # Read SQLite directly before any constructor can recover the journal.
        store = ExperimentStore(directory / "recovery.sqlite")
        with store.connection() as db:
            before = {"result_ready": db.execute("SELECT count(*) FROM trials WHERE status='result_ready'").fetchone()[0],
                      "reward_claims": db.execute("SELECT count(*) FROM reward_claims").fetchone()[0]}
        if before != {"result_ready": 1, "reward_claims": 0}:
            raise AssertionError(before)
        resumed = subprocess.run(command + ["--_phase", "resume"], capture_output=True, text=True, timeout=60)
        if resumed.returncode:
            raise RuntimeError(resumed.stderr[-3000:])
        cases[boundary] = {"crashed_exit_code": crash.returncode, "before_recovery": before,
                           "recovered": read_json(directory / "resumed.json")}
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in source_hashes.items()):
        raise AssertionError("Original evidence was modified")
    receipt = {"schema_version": 1, "ok": True,
               "scope": "Whole-process crash/restart and deduplication of archived real CUDA evidence; no new performance observation",
               "archived_evaluation_id": fixture["result"]["evaluation_id"],
               "source_evidence_sha256": source_hashes, "cases": cases,
               "execution_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "model_calls": 0, "gpu_requests": 0, "original_evidence_unchanged": True}
    write_json(output / "process-recovery-acceptance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_directory", type=Path)
    parser.add_argument("--_phase", choices=("after_result", "after_tell", "resume"))
    args = parser.parse_args()
    if args._phase:
        child(args._directory, args._phase)
        return
    if args.campaign is None or args.output is None:
        parser.error("--campaign and --output are required")
    receipt = run_acceptance(args.campaign, args.output)
    print(json.dumps({"ok": receipt["ok"], "receipt": str(args.output / "process-recovery-acceptance.json"),
                      "model_calls": 0, "gpu_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
