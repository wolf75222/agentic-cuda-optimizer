#!/usr/bin/env python3
"""Predeclared functional legacy/exchange transport comparison; no gain claim.

Default only writes a CPU plan. --run owns one fresh Slurm allocation, checks
the same CPU-oracle GEMM requests through both transports in a fixed order,
and records RPC wall time separately from CUDA Events and compilation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import SlurmBackend, SlurmConfig, device_identity, write_json
from scripts.gpu_contract_acceptance import prepare
from scripts.acceptance_intent import persist_allocation_intent


def protocol():
    order = []
    for repetition in range(2):
        for index, size in enumerate((17, 63)):
            transports = ("legacy", "exchange") if (repetition + index) % 2 == 0 else ("exchange", "legacy")
            for mode in ("run", "benchmark"):
                for transport in transports:
                    order.append({"index": len(order), "repetition": repetition, "n": size,
                                  "mode": mode, "transport": transport})
    return {"schema_version": 1, "kind": "functional_rpc_acceptance", "order": order,
            "workload": "matmul", "kernel": "matmul", "precision": "float32", "layout": "row-major",
            "source_sha256": hashlib.sha256((REPO / "examples/naive_matmul.cu").read_bytes()).hexdigest(),
            "case_seeds": {"17": 1730, "63": 1731}, "compiler_options": [],
            "launch": {"block": [16, 16, 1], "grid": "ceil(n/16),ceil(n/16),1", "shared_memory_bytes": 0},
            "prewarm": {"transport": "exchange", "mode": "run", "n": 17},
            "warmup_per_benchmark": 1, "launches_per_benchmark": 2, "series_per_request": 1,
            "expected_requests_per_transport": 8, "expected_benchmark_requests_per_transport": 4,
            "expected_raw_timings_per_transport": 8, "model_calls": 0,
            "ssh_count_method": "Count controller SSHTransport.run invocations during each request, including retries",
            "scope": "Transport functionality and measured RPC cost; no kernel speedup or statistical superiority claim"}


def provenance(execution_config=REPO / "configs/romeo.json"):
    files = ["optimizer_agent/execution_backend.py", "optimizer_agent/remote_worker.py",
             "optimizer_agent/evaluation.py", "optimizer_agent/campaign.py", "optimizer_agent/campaign_config.py", "optimizer_agent/benchmark_workloads.py",
             "optimizer_agent/compile_cache.py", "optimizer_agent/use_harness.py", "scripts/gpu_rpc_acceptance.py",
             "scripts/gpu_contract_acceptance.py", "scripts/acceptance_intent.py", "examples/naive_matmul.cu"]
    files += [str(p.relative_to(REPO)) for p in (REPO / "cuda_test_harness").rglob("*")
              if p.is_file() and p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".txt")
              and not any(part.startswith(("build", "cmake-build")) for part in p.relative_to(REPO).parts)]
    paths = [REPO / name for name in files] + [Path(execution_config).resolve()]
    return {"HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines(),
            "sources": {str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path):
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def run_checks(backend, directory, run_id, plan):
    directory = Path(directory)
    original_config = backend.config
    backend.config = replace(original_config, transport_mode="exchange")
    fast = backend.runner()
    legacy = backend.runner()
    evaluator, candidate, _, cases, _ = prepare(directory / "fixed-inputs", run_id,
        runner=fast, properties=fast.get_device_properties(0))
    cases = {case.metadata["n"]: case for case in cases}
    if candidate._source_hash != plan["source_sha256"] or any(cases[int(n)].seed != seed for n, seed in plan["case_seeds"].items()):
        raise RuntimeError("Actual source/inputs differ from the predeclared protocol")
    properties = device_identity(fast.get_device_properties(0))
    rows = []
    original_run = backend.transport.run
    ssh_calls = 0
    def counted_ssh(*args, **kwargs):
        nonlocal ssh_calls
        ssh_calls += 1
        return original_run(*args, **kwargs)
    backend.transport.run = counted_ssh
    try:
        warmup_request = evaluator._request(candidate, cases[17])
        warmup = fast.run(run_id + "-prewarm", warmup_request)
        if not warmup.get("ok") or not evaluator._compare(cases[17], warmup, evaluator.oracle(cases[17]))["ok"]:
            raise RuntimeError("Prewarm failed independent CPU oracle")
        write_json(directory / "prewarm.json", {"request": asdict(warmup_request), "report": warmup})
        for item in plan["order"]:
            backend.config = replace(original_config, transport_mode=item["transport"])
            runner = fast if item["transport"] == "exchange" else legacy
            case = cases[item["n"]]
            request = evaluator._request(candidate, case, benchmark=item["mode"] == "benchmark")
            started = time.monotonic()
            before_ssh = ssh_calls
            report = runner.run(run_id + "-" + str(item["index"]), request)
            wall_seconds = time.monotonic() - started
            comparison = evaluator._compare(case, report, evaluator.oracle(case)) if report.get("ok") else {"ok": False}
            expected_count = plan["launches_per_benchmark"] if item["mode"] == "benchmark" else 0
            native_protocol = report.get("measurement_protocol", {})
            protocol_matches = (native_protocol.get("warmup") == request.warmup
                and native_protocol.get("iterations") == request.iterations and native_protocol.get("series") == request.series)
            row = {**item, "request": asdict(request), "report": report, "oracle": comparison,
                   "controller_wall_seconds": wall_seconds,
                   "ssh_processes": ssh_calls - before_ssh,
                   "raw_timing_count": len(report.get("latency_samples_ms", ())),
                   "native_protocol_matches": protocol_matches,
                   "ok": bool(report.get("ok") and comparison["ok"] and evaluator._check_environment(report)
                       and report.get("guard_readonly_ok") is True and report.get("guard_outputs_poisoned") is True
                       and protocol_matches and len(report.get("latency_samples_ms", ())) == expected_count)}
            rows.append(row)
            write_json(directory / "request-receipts" / (str(item["index"]).zfill(2) + ".json"), row)
            if not row["ok"]:
                raise RuntimeError("Functional transport request failed: " + str(item["index"]))
    finally:
        backend.config = original_config
        backend.transport.run = original_run
    modes = {}
    for mode in ("legacy", "exchange"):
        selected = [row for row in rows if row["transport"] == mode]
        modes[mode] = {"requests": len(selected), "benchmark_requests": sum(row["mode"] == "benchmark" for row in selected),
            "raw_timings": sum(row["raw_timing_count"] for row in selected),
            "median_controller_wall_seconds": statistics.median(row["controller_wall_seconds"] for row in selected),
            "total_controller_wall_seconds": sum(row["controller_wall_seconds"] for row in selected),
            "native_reservation_seconds": sum(row["report"]["reservation_wall_seconds"] for row in selected),
            "native_compile_seconds": sum(row["report"]["compile_ms"] / 1000 for row in selected),
            "blob_upload_bytes": sum(row["report"]["transport_receipt"].get("blob_upload_bytes", 0) for row in selected),
            "ssh_processes": [row["ssh_processes"] for row in selected],
            "ssh_roundtrips": [row["report"]["transport_receipt"].get("ssh_roundtrips") for row in selected]}
    checks = {"fixed_order_preserved": [{key: row[key] for key in ("index", "repetition", "n", "mode", "transport")} for row in rows] == plan["order"],
        "native_execution_order_preserved": all(isinstance(row["report"].get("started_at_unix_ms"), int) for row in rows)
            and [row["report"]["started_at_unix_ms"] for row in rows] == sorted(row["report"]["started_at_unix_ms"] for row in rows),
        "all_cpu_oracles_and_guards_passed": all(row["ok"] for row in rows),
        "same_hardware_toolchain": all(device_identity(row["report"]["environment"]) == properties for row in rows),
        "request_and_raw_counts_unchanged": all(mode["requests"] == plan["expected_requests_per_transport"]
            and mode["benchmark_requests"] == plan["expected_benchmark_requests_per_transport"]
            and mode["raw_timings"] == plan["expected_raw_timings_per_transport"] for mode in modes.values()),
        "fast_path_used_one_rpc_per_request": all(value == 1 for value in modes["exchange"]["ssh_roundtrips"]),
        "fast_rpc_count_matches_actual_ssh_processes": modes["exchange"]["ssh_processes"] == modes["exchange"]["ssh_roundtrips"],
        "cached_blobs_avoided_repeated_transfers": sum(row["report"]["transport_receipt"].get("blob_uploads", 0)
            for row in rows if row["transport"] == "exchange") <= 2}
    return {"ok": all(checks.values()), "checks": checks, "hardware_identity": properties,
            "modes": modes, "requests": rows, "scope": plan["scope"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="rpc-acceptance-" + uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    args = parser.parse_args(argv)
    directory = args.output.resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", "allocation-receipt.json", "worker/slurm-session.json")):
        raise ValueError("Use a fresh directory; existing evidence/allocation is never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    plan = protocol()
    write_json(directory / "protocol.json", plan)
    if not args.run:
        print(json.dumps({"submitted": False, "protocol": str(directory / "protocol.json")}))
        return 0
    backend = SlurmBackend(SlurmConfig.load(args.execution_config), REPO, directory / "worker",
        session_id="rpc-acceptance-" + uuid.uuid4().hex)
    receipt = {"ok": False, "job_id": None, "model_calls": 0, "session_id": backend.session_id,
               "remote_dir": backend.remote_dir, "source_provenance": provenance(args.execution_config)}
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event": "rpc_acceptance_started", "job_id": backend.job_id, "session_id": backend.session_id}), flush=True)
        backend.wait_until_ready()
        receipt["functional"] = run_checks(backend, directory, args.run_id, plan)
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if backend.job_id:
            try:
                backend.close()
                receipt["terminal"] = backend.wait_until_finished(timeout=60)
                if not receipt["terminal"].get("terminal"):
                    backend.close(cancel=True)
                    receipt["terminal"] = backend.wait_until_finished(timeout=60)
            except Exception as error:
                receipt["cleanup_error"] = str(error)
        receipt["source_provenance_after"] = provenance(args.execution_config)
        unchanged = receipt["source_provenance"]["sources"] == receipt["source_provenance_after"]["sources"]
        ended = receipt.get("terminal", {})
        receipt["ok"] = bool(receipt.get("functional", {}).get("ok") and unchanged and not receipt.get("error")
            and not receipt.get("cleanup_error") and ended.get("terminal") and ended.get("state") == "COMPLETED" and ended.get("exit_code") == "0:0")
        write_json(directory / "allocation-receipt.json", receipt)
    print(json.dumps({"ok": receipt["ok"], "job_id": receipt["job_id"], "receipt": str(directory / "allocation-receipt.json")}))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
