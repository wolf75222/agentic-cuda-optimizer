#!/usr/bin/env python3
"""Prepare/run scoped AC-09, AC-13 and AC-20 checks without a model.

Default: CPU-only preparation. --run owns one new Slurm allocation. Precision
violations never reach CUDA; a deliberately incompatible cache receipt belongs
only to this allocation. No input mutation or damaged-context probe runs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
import uuid
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig, Candidate, Evaluator, LaunchRule, SlurmBackend, SlurmConfig,
    WorkloadContract, WorkloadSuite, device_identity, write_json,
)
from optimizer_agent.compile_cache import compilation_manifest
from optimizer_agent.autotuning import Autotuner
from scripts.gpu_autotuning_acceptance import ReceiptRunner
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent
from use_harness import KernelRequest


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def prepare(directory, run_id, *, runner=None, properties=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(run_id=run_id, workload="matmul", experimental_mode="test",
        research_required=False, web_search_required=False, search_sizes=(17, 63),
        final_sizes=(17,), benchmark_series=2, launches_per_series=2, warmup=1,
        confirmation_batches=2, required_sanitizers=(), profile=False)
    suite = WorkloadSuite("matmul", directory / "inputs", config)
    cases = suite.cases(final=False)
    contract = WorkloadContract(run_id, 1, "matmul", suite.signature,
        "Independent NumPy float64 matrix multiplication with float32 outputs",
        benchmark_series=2, launches_per_series=2, warmup=1, confirmation_batches=2,
        allowed_precision="float32", metadata_json=json.dumps({"precision": "float32",
            "target_environment": device_identity(properties or {}),
            "scope": "Functional contract/cache/reconstruction checks; no optimization gain claim"}))
    evaluator = Evaluator(runner, contract, suite.oracle,
        {"short": cases[:2], "search": cases, "performance": cases[1:]})
    baseline = suite.baseline(REPO / "examples/naive_matmul.cu")
    text = baseline.source.read_text()
    variants = {}
    for name, source in {
        "double": text.replace("float sum = 0.0f", "double sum = 0.0"),
        "tf32": text.replace("a[row*n + k]", "__float_to_tf32(a[row*n + k])"),
        "half": text.replace("float sum = 0.0f;", "__half forbidden_precision; float sum = 0.0f;"),
    }.items():
        path = directory / "precision" / (name + ".cu")
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        variants[name] = Candidate("precision-" + name, path, "matmul", baseline.launch_rule)
    plan = {"run_id": run_id, "submitted": False, "model_calls": 0,
        "planned_checks": ["AC09 explicit double/TF32/half rejected before any harness call",
            "AC13 actual native cache hit/miss plus isolated incompatible toolchain-tag rejection",
            "AC13 local dependency-content manifest invalidation and remote dependency fail-closed",
            "AC20 exact source/launch/options/inputs bundle reconstruction and independent CUDA correctness",
            "AC12 additional domain [32,2048]: one actual valid launch and one hardware-filtered launch"],
        "destructive_job_needed": False,
        "limitations": ["No actual driver/toolkit upgrade is performed",
                        "External dependencies are not staged by the remote transport; no GPU-header-recompilation claim"],
        "contract": asdict(contract), "contract_hash": evaluator.contract_hash,
        "helper_sha256": sha256(Path(__file__).read_bytes())}
    write_json(directory / "plan.json", plan)
    return evaluator, baseline, variants, cases, plan


def toolchain(properties):
    return {key: properties[key] for key in ("driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")}


def bind_cache(request, properties):
    manifest = compilation_manifest(Path(request.source), options=request.compiler_options,
        architecture=properties["architecture"], toolchain=toolchain(properties))
    return replace(request, compile_cache_key=manifest["key"]), manifest


def create_bundle(path, candidate, request, properties):
    payload = asdict(request)
    files = {"source.cu": candidate.source.read_bytes()}
    payload.update(source="source.cu", cache_dir=None, compile_cache_key=None)
    for index, argument in enumerate(payload["arguments"]):
        if argument.get("input_file"):
            name = "inputs/argument-" + str(index) + ".bin"
            files[name] = Path(argument["input_file"]).read_bytes()
            argument["input_file"] = name
    files["request.json"] = encoded(payload)
    files["launch.json"] = encoded({"launch_rule": asdict(candidate.launch_rule), "parameters": candidate.parameters})
    manifest = {"schema_version": 1, "candidate_id": candidate.candidate_id,
        "artifact_hash": candidate.artifact_hash, "source_hash": candidate._source_hash,
        "kernel": candidate.kernel, "launch_rule": asdict(candidate.launch_rule), "parameters": candidate.parameters,
        "compiler_options": request.compiler_options, "options_sha256": sha256(encoded(request.compiler_options)),
        "architecture": properties["architecture"], "toolchain": toolchain(properties),
        "files": {name: sha256(data) for name, data in files.items()}}
    manifest = json.loads(encoded(manifest))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", encoded(manifest))
    return manifest


def reconstruct_bundle(path, directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Bundle reconstruction requires a fresh output directory")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if set(archive.namelist()) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Unexpected or missing bundle entries")
        payloads = {}
        for name, expected_hash in manifest["files"].items():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("Unsafe bundle path")
            value = archive.read(name)
            if sha256(value) != expected_hash:
                raise ValueError("Bundle file checksum mismatch: " + name)
            payloads[name] = value
    if sha256(payloads["source.cu"]) != manifest["source_hash"]:
        raise ValueError("Bundle source identity changed")
    request = json.loads(payloads["request.json"])
    launch = json.loads(payloads["launch.json"])
    if (request["compiler_options"] != manifest["compiler_options"]
            or request["kernel"] != manifest["kernel"]
            or sha256(encoded(request["compiler_options"])) != manifest["options_sha256"]
            or launch != {"launch_rule": manifest["launch_rule"], "parameters": manifest["parameters"]}):
        raise ValueError("Bundle launch/options identity changed")
    directory.mkdir(parents=True)
    for name, value in payloads.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    candidate = Candidate(manifest["candidate_id"], directory / "source.cu", manifest["kernel"],
                          LaunchRule(**manifest["launch_rule"]), manifest["parameters"])
    if candidate.artifact_hash != manifest["artifact_hash"]:
        raise ValueError("Reconstructed artifact does not match source/launch/parameters")
    request["source"] = str(candidate.source)
    for argument in request["arguments"]:
        if argument.get("input_file"):
            target = (directory / argument["input_file"]).resolve()
            if not target.is_relative_to(directory) or argument["input_file"] not in payloads:
                raise ValueError("Request input escaped the verified bundle")
            argument["input_file"] = str(target)
    return candidate, KernelRequest.from_dict(request), manifest


def invalid_launch_acceptance(evaluator, baseline, recording, directory, properties):
    """Cover the missing invalid configuration without repeating valid AC-12."""
    template = {"source": baseline.source.read_text(), "kernel": baseline.kernel,
        "launch_rule": {**asdict(baseline.launch_rule), "block": [16, 8, 1]},
        "parameter_bindings": [{"parameter": "BLOCK_X", "target": "block_x"}]}
    start = len(recording.calls)
    tuner = Autotuner(evaluator, work_directory=Path(directory) / "invalid-launch/candidates")
    result = tuner.search(template, {"BLOCK_X": [32, 2048]}, {"configurations": 2, "compilations": 2},
                          hardware_limits=properties)
    observations = [asdict(item) for item in result.configurations]
    calls = recording.calls[start:]
    measured = [row for row in observations if row["status"] == "measured"]
    filtered = [row for row in observations if row["status"] == "filtered"]
    return {"domain": {"BLOCK_X": [32, 2048]}, "observations": observations,
        "winner_remeasurement": result.remeasurement,
        "counters": {"measured_configurations": len(measured), "filtered_configurations": len(filtered),
                     "native_requests": len(calls),
                     "native_requests_with_2048_threads_x": sum(row["request"]["block"][0] == 2048 for row in calls)},
        "ok": (len(measured) == len(filtered) == 1 and measured[0]["parameters"] == {"BLOCK_X": 32}
               and filtered[0]["parameters"] == {"BLOCK_X": 2048}
               and not filtered[0].get("measurement") and bool(calls)
               and all(row["request"]["block"][0] == 32 for row in calls)
               and result.remeasurement is not None and result.remeasurement.get("ok") is True
               and result.remeasurement["evaluation_id"] != measured[0]["measurement"]["evaluation_id"]),
        "scope": "Functional valid/invalid domain coverage only; no cross-allocation gain comparison"}


def run_checks(runner, directory, run_id):
    directory = Path(directory).resolve()
    recording = ReceiptRunner(runner, directory)
    properties = runner.get_device_properties(0)
    evaluator, baseline, variants, cases, plan = prepare(directory, run_id, runner=recording, properties=properties)
    evidence = {"kind": "scoped_gpu_contract_acceptance", "checks": {}, "scope": plan["limitations"],
                "hardware_identity": device_identity(properties), "precision": {}, "model_calls": 0}
    control = evaluator.benchmark(baseline)
    evidence["control"] = control
    evidence["checks"]["valid_control_measured"] = control.get("ok") is True
    if not control.get("ok"):
        raise RuntimeError("The valid control failed before negative acceptance probes")
    for name, candidate in variants.items():
        before = len(recording.calls)
        result = evaluator.benchmark(candidate)
        evidence["precision"][name] = result
        evidence["checks"]["AC09_" + name + "_rejected_before_harness"] = (
            result.get("status") == "contract_violation" and not result.get("ok") and len(recording.calls) == before)

    request, manifest = bind_cache(replace(evaluator._request(baseline, cases[-1]), compiler_options=["--fmad=true"]), properties)
    reports = [recording.run(uuid.uuid4().hex, request), recording.run(uuid.uuid4().hex, request)]
    backend = runner.backend
    cache_path = backend.remote_dir + "/compile-cache/" + manifest["key"] + ".cubin.json"
    before_bytes = backend.transport.read(cache_path)
    bad_receipt = json.loads(before_bytes)
    if bad_receipt["identity"]["key"] != manifest["key"]:
        raise ValueError("The cache entry does not belong to this experiment")
    bad_receipt["identity"]["driver_version"] += 1
    backend.transport.write(cache_path, encoded(bad_receipt))
    reports += [recording.run(uuid.uuid4().hex, request), recording.run(uuid.uuid4().hex, request)]
    evidence["native_cache"] = {"reports": reports, "before_receipt_sha256": sha256(before_bytes),
        "incompatible_receipt": bad_receipt, "repaired_receipt": json.loads(backend.transport.read(cache_path)),
        "scope": "Cache identity tag deliberately differs; actual compiler, toolkit and driver remain unchanged"}
    expected = evaluator.oracle(cases[-1])
    evidence["checks"]["AC13_native_toolchain_tag_mismatch_recompiled"] = (
        [item.get("compile_cache_hit") for item in reports] == [False, True, False, True]
        and all(item.get("ok") and evaluator._compare(cases[-1], item, expected)["ok"] for item in reports))

    dependency = directory / "dependency-policy/header.h"
    dependency.parent.mkdir()
    dependency.write_text("#define VALUE 1\n")
    source = dependency.with_name("source.cu")
    source.write_text('#include "header.h"\n' + baseline.source.read_text())
    first = compilation_manifest(source, architecture=properties["architecture"], toolchain=toolchain(properties))
    dependency.write_text("#define VALUE 2\n")
    second = compilation_manifest(source, architecture=properties["architecture"], toolchain=toolchain(properties))
    changed_toolchain = {**toolchain(properties), "nvrtc_version": str(properties["nvrtc_version"]) + "-policy-test"}
    third = compilation_manifest(source, architecture=properties["architecture"], toolchain=changed_toolchain)
    rejected = False
    try:
        runner.run(uuid.uuid4().hex, replace(request, dependencies=[str(dependency)]))
    except ValueError as error:
        rejected = "dependenc" in str(error).lower()
        evidence["dependency_transport_error"] = str(error)
    evidence["dependency_policy_manifests"] = [first, second, third]
    evidence["checks"]["AC13_dependency_content_manifest_invalidated"] = first["key"] != second["key"]
    evidence["checks"]["AC13_toolchain_manifest_invalidated"] = second["key"] != third["key"]
    evidence["checks"]["unstaged_dependencies_fail_closed"] = rejected

    bundle = directory / "exact-candidate.zip"
    original_manifest = create_bundle(bundle, baseline, request, properties)
    rebuilt, rebuilt_request, bundle_manifest = reconstruct_bundle(bundle, directory / "reconstructed")
    rebuilt_request, rebuilt_manifest = bind_cache(rebuilt_request, properties)
    actual = recording.run(uuid.uuid4().hex, rebuilt_request)
    evidence["reconstruction"] = {"bundle_sha256": sha256(bundle.read_bytes()), "bundle_manifest": bundle_manifest,
        "original_compile_manifest": manifest, "reconstructed_compile_manifest": rebuilt_manifest,
        "reconstructed_artifact_hash": rebuilt.artifact_hash, "report": actual,
        "cache_key_note": "Compilation manifest paths change under reconstruction; source/launch/options/input bytes and target identity do not"}
    evidence["checks"]["AC20_exact_bundle_reconstructed_and_executed"] = (
        rebuilt.artifact_hash == baseline.artifact_hash and original_manifest == bundle_manifest
        and rebuilt_request.compiler_options == request.compiler_options
        and actual.get("ok") and evaluator._compare(cases[-1], actual, expected)["ok"]
        and actual.get("compilation_identity", {}).get("options") == request.compiler_options)
    invalid_launch = invalid_launch_acceptance(evaluator, baseline, recording, directory, properties)
    evidence["invalid_launch_tuning"] = invalid_launch
    evidence["checks"]["AC12_valid_measured_invalid_hardware_filtered"] = invalid_launch["ok"]
    evidence["requests"] = recording.calls
    evidence["ok"] = all(evidence["checks"].values())
    write_json(directory / "contract-acceptance.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, default=REPO / "configs/romeo.json")
    parser.add_argument("--run-id", default="contract-acceptance-" + uuid.uuid4().hex[:10])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    directory = args.output.resolve()
    if any((directory / name).exists() for name in ("allocation-intent.json", "allocation-receipt.json")):
        raise ValueError("Use a fresh experiment directory; reconcile the recorded allocation intent/receipt")
    if not args.run:
        prepare(directory, args.run_id)
        print(json.dumps({"submitted": False, "plan": str(directory / "plan.json")}))
        return 0
    if (directory / "worker/slurm-session.json").exists() or (directory / "contract-acceptance.json").exists():
        raise ValueError("Use a fresh experiment directory, never overwrite/reuse an existing allocation")
    backend = SlurmBackend(SlurmConfig.load(args.execution_config), REPO, directory / "worker",
                           session_id="contract-probes-" + uuid.uuid4().hex)
    receipt = {"ok": False, "job_id": None,
        "controller_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()}
    receipt.update(session_id=backend.session_id, remote_dir=backend.remote_dir)
    receipt["source_provenance"] = controller_provenance(REPO, __file__, args.execution_config)
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    try:
        receipt["job_id"] = backend.start()
        backend.wait_until_ready()
        receipt["checks"] = run_checks(backend.runner(), directory, args.run_id)
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
        receipt["ok"] = bool(receipt.get("checks", {}).get("ok") and receipt.get("terminal", {}).get("terminal")
            and not receipt.get("cleanup_error")
            and receipt["terminal"].get("state") == "COMPLETED" and receipt["terminal"].get("exit_code") == "0:0")
        write_json(directory / "allocation-receipt.json", receipt)
    print(json.dumps({"ok": receipt["ok"], "job_id": receipt["job_id"], "receipt": str(directory / "allocation-receipt.json")}))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
