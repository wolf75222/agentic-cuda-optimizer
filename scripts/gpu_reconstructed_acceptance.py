#!/usr/bin/env python3
"""AC20: one archived native finalist request, independent CPU oracle, no score.

Default reconstructs and writes a local plan. --run owns one fresh allocation
and executes exactly one archived mode=run request; no benchmark/model calls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback
import uuid
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent import portable_export
from optimizer_agent.campaign import SlurmBackend, SlurmConfig, device_identity, write_json
from optimizer_agent.evaluation import _source_contract_violations
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent

ARTIFACT = "4c12a3216027cb3c44b95446ca4574e53feb61836d6a30bf4cf6317aad7ba099"
ABI = 'extern "C" __global__ void matmul(const float* a, const float* b, float* c, int n)'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_identity(request):
    data = asdict(request)
    data.pop("cache_dir", None)
    data.pop("compile_cache_key", None)
    data["source_sha256"] = sha(data.pop("source"))
    for item in data["arguments"]:
        if item.get("input_file"):
            item["input_sha256"] = sha(item.pop("input_file"))
    data["dependencies"] = [sha(path) for path in data.get("dependencies", [])]
    return data


def prepare(directory, bundle, bundle_sha256, *, selection="comparison.json#native_codex", request_index=3,
            expected_artifact=ARTIFACT, origin_comparison=None, reconstruction_name="plan-reconstruction"):
    directory, bundle = Path(directory).resolve(), Path(bundle).resolve()
    if sha(bundle) != bundle_sha256:
        raise ValueError("Bundle SHA256 differs from the predeclared archive")
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
        manifest = json.loads(archive.read("export-manifest.json"))
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"export-manifest.json"}:
            raise ValueError("Duplicate, missing or unexpected archive members")
        for name, identity in manifest["files"].items():
            data = archive.read(name)
            if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError("Archive member SHA256 differs: " + name)
        original_bytes = archive.read("evidence/comparison.json")
        original = json.loads(original_bytes)
    candidate, requests, _ = portable_export.reconstruct_candidate(bundle, selection, directory / reconstruction_name)
    if candidate.artifact_hash != expected_artifact or candidate.kernel != "matmul":
        raise ValueError("Reconstructed candidate differs from the declared native finalist")
    archived = original["candidates"]["native_codex"]
    if (archived["artifact_hash"] != expected_artifact or archived["source_hash"] != candidate._source_hash
            or archived["launch_rule"] != json.loads(json.dumps(asdict(candidate.launch_rule)))
            or archived["parameters"] != dict(candidate.parameters)):
        raise ValueError("Archived comparison and reconstructed candidate disagree")
    if type(request_index) is not int or not 0 <= request_index < len(requests):
        raise ValueError("Archived request index is unavailable")
    request = requests[request_index]
    args = asdict(request)["arguments"]
    if (request.mode != "run" or request.kernel != "matmul" or request.device != 0
            or request.warmup != 0 or request.iterations != 1 or request.series != 1
            or not request.guard_outputs or request.sentinel_byte not in (127, 255)
            or request.dependencies or len(args) != 4):
        raise ValueError("Selected request must be one guarded archived run, without benchmarks/dependencies")
    if ([(v.get("kind"), str(v.get("direction"))) for v in args[:3]] !=
            [("buffer", "input"), ("buffer", "input"), ("buffer", "output")]
            or args[3] != {"type": "int32", "value": 316, "kind": "scalar"}
            or any(v["bytes"] != 316 * 316 * 4 for v in args[:3]) or args[2]["input_file"] is not None
            or len({str(Path(v["input_file"]).resolve()) for v in args[:2]}) != 2):
        raise ValueError("Archived request does not match the disjoint A/B/C float32 GEMM ABI at n=316")
    launch = candidate.launch_rule.apply({"n": 316})
    if any(getattr(request, key) != value for key, value in launch.items()):
        raise ValueError("Request launch differs from the frozen candidate")
    violations = _source_contract_violations(candidate.source.read_text(), candidate.kernel, ABI, "float32")
    if violations:
        raise ValueError("Reconstructed source violates ABI/precision: " + "; ".join(violations))
    comparison_sha = hashlib.sha256(original_bytes).hexdigest()
    if origin_comparison is not None and sha(origin_comparison) != comparison_sha:
        raise ValueError("Original comparison bytes differ from the archived evidence")
    a, b = [np.fromfile(arg.input_file, dtype="<f4").reshape(316, 316) for arg in request.arguments[:2]]
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype("<f4")
    if not np.isfinite(expected).all():
        raise ValueError("Independent CPU oracle produced nonfinite values")
    plan = {"kind": "reconstructed_native_finalist_acceptance", "artifact_hash": candidate.artifact_hash,
        "source_sha256": candidate._source_hash, "bundle": str(bundle), "bundle_sha256": bundle_sha256,
        "selection": selection, "request_index": request_index, "request": asdict(request),
        "request_identity": request_identity(request), "launch_rule": asdict(candidate.launch_rule),
        "parameters": dict(candidate.parameters), "ABI": ABI, "n": 316, "precision": "float32",
        "oracle": {"implementation": "NumPy float64 A @ B cast to float32", "numpy_version": np.__version__,
                   "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(), "rtol": 2e-4, "atol": 2e-4},
        "original_comparison_sha256": comparison_sha, "original_contract_hash": original.get("contract_hash"),
        "original_environment": original.get("hardware_identity"), "origin_comparison": str(Path(origin_comparison).resolve()) if origin_comparison else None,
        "export_status": manifest["status"], "portable_module_sha256": sha(portable_export.__file__),
        "harness_requests": 1, "benchmark_requests": 0, "model_calls": 0, "new_scores_created": False,
        "scope": "One relocated archived execution with independent numerical verification; original qualification and comparison scores are not inherited or updated"}
    write_json(directory / "plan.json", plan)
    return candidate, request, expected, plan


def execute_one(runner, candidate, request, expected, plan, run_id):
    before = request_identity(request)
    properties = runner.get_device_properties(0)
    if not properties.get("uuid"):
        raise ValueError("Actual allocated GPU UUID is unavailable")
    report = runner.run(run_id, request)
    outputs = report.get("outputs", [])
    actual = np.fromfile(outputs[0]["path"], dtype="<f4") if len(outputs) == 1 else np.array([], dtype="<f4")
    numerical = actual.size == expected.size and bool(np.isfinite(actual).all()) and bool(np.all(np.isclose(
        actual.astype(np.float64), expected.ravel().astype(np.float64), rtol=plan["oracle"]["rtol"], atol=plan["oracle"]["atol"])))
    environment = report.get("environment", {})
    keys = ("uuid", "architecture", "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
    native_protocol = report.get("measurement_protocol", {})
    compilation = report.get("compilation_identity", {})
    checks = {"independent_cpu_oracle": numerical, "native_success": report.get("ok") is True,
        "actual_gpu_identity": all(key in properties and environment.get(key) == properties[key] for key in keys),
        "read_only_inputs": report.get("guard_readonly_ok") is True,
        "missing_output_sentinels": report.get("guard_outputs_poisoned") is True,
        "source_launch_options_inputs_unchanged": before == plan["request_identity"] == request_identity(request),
        "exact_source_and_compiler_options_executed": compilation.get("source") == candidate.source.read_text()
            and compilation.get("options") == request.compiler_options,
        "one_run_without_timing": report.get("latency_samples_ms") == [] and native_protocol.get("warmup") == 0
            and native_protocol.get("iterations") == native_protocol.get("series") == 1,
        "clean_worker": report.get("worker_reusable") is True and not report.get("worker_poisoned")}
    return {"ok": all(checks.values()), "checks": checks, "report": report,
        "actual_environment": device_identity(properties), "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "max_absolute_error": float(np.max(np.abs(actual.astype(np.float64) - expected.ravel().astype(np.float64)))) if actual.size == expected.size and np.isfinite(actual).all() else None,
        "harness_requests": 1, "benchmark_requests": 0, "new_scores_created": False}


def run_acceptance(directory, bundle, bundle_sha256, execution_config, *, run_id, selection, request_index,
                   expected_artifact=ARTIFACT, origin_comparison=None, backend_factory=None):
    directory = Path(directory).resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", "allocation-receipt.json", "worker/slurm-session.json")):
        raise ValueError("Use a fresh directory; reconcile the existing allocation intent")
    candidate, request, expected, plan = prepare(directory, bundle, bundle_sha256, selection=selection,
        request_index=request_index, expected_artifact=expected_artifact, origin_comparison=origin_comparison,
        reconstruction_name="execution-reconstruction")
    backend = (backend_factory or SlurmBackend)(SlurmConfig.load(execution_config), REPO, directory / "worker",
        session_id="reconstructed-native-" + uuid.uuid4().hex)
    receipt = {"ok": False, "job_id": None, "session_id": backend.session_id, "remote_dir": backend.remote_dir,
        "plan": plan, "model_calls": 0, "new_scores_created": False,
        "evidence_kind": "simulated_cpu_test" if getattr(backend, "simulation", False) else "romeo_execution",
        "source_provenance": controller_provenance(REPO, __file__, execution_config)}
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event": "reconstructed_acceptance_started", "job_id": backend.job_id, "session_id": backend.session_id}), flush=True)
        backend.wait_until_ready()
        receipt["execution"] = execute_one(backend.runner(), candidate, request, expected, plan, run_id)
        receipt["execution"]["checks"]["owned_slurm_job"] = str(receipt["execution"]["report"].get("slurm_job_id")) == str(backend.job_id)
        receipt["execution"]["ok"] = all(receipt["execution"]["checks"].values())
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
        receipt["source_provenance_after"] = controller_provenance(REPO, __file__, execution_config)
        receipt["bundle_unchanged"] = sha(bundle) == bundle_sha256
        receipt["origin_comparison_unchanged"] = origin_comparison is None or sha(origin_comparison) == plan["original_comparison_sha256"]
        ended = receipt.get("terminal", {})
        receipt["ok"] = bool(receipt.get("execution", {}).get("ok") and not receipt.get("error") and not receipt.get("cleanup_error")
            and receipt["bundle_unchanged"] and receipt["origin_comparison_unchanged"]
            and receipt["source_provenance"]["sources"] == receipt["source_provenance_after"]["sources"]
            and ended.get("terminal") and ended.get("state") == "COMPLETED" and ended.get("exit_code") == "0:0"
            and str(ended.get("job_id")) == str(receipt["job_id"]))
        write_json(directory / "allocation-receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", default="comparison.json#native_codex")
    parser.add_argument("--request-index", type=int, default=3)
    parser.add_argument("--artifact-hash", default=ARTIFACT)
    parser.add_argument("--origin-comparison", type=Path)
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    parser.add_argument("--run-id", default="reconstructed-" + uuid.uuid4().hex[:12])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if any((args.output / name).exists() for name in ("allocation-intent.json", "allocation-receipt.json")):
        raise ValueError("Use a fresh directory; existing allocation intent/evidence is preserved")
    parameters = dict(selection=args.selection, request_index=args.request_index,
                      expected_artifact=args.artifact_hash, origin_comparison=args.origin_comparison)
    if not args.run:
        prepare(args.output, args.bundle, args.bundle_sha256, **parameters)
        print(json.dumps({"submitted": False, "plan": str(args.output.resolve() / "plan.json")}))
        return 0
    result = run_acceptance(args.output, args.bundle, args.bundle_sha256, args.execution_config, run_id=args.run_id, **parameters)
    print(json.dumps({"ok": result["ok"], "job_id": result["job_id"], "receipt": str(args.output.resolve() / "allocation-receipt.json")}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
