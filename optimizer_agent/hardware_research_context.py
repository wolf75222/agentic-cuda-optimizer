"""Authenticated allocation metadata for research, separate from timing identity."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def _uuid(value):
    value = str(value).removeprefix("GPU-").replace("-", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Hardware research context requires an unambiguous physical GPU UUID")
    return value


def archived_hardware_research_context(worker_directory, environment, *, expected_job_id,
                                      expected_session_id, expected_remote_dir=None,
                                      expected_hardware=None, expected_receipt_hashes=None):
    """Read only the fixed hardware/ownership receipts, never infer compatibility.

    An allocation-start observation remains historical even if its worker is
    still running. This annex does not alter a contract or environment hash.
    """
    directory = Path(worker_directory).resolve()
    records, references = {}, {}
    for name in ("hardware.json", "slurm-session.json"):
        path = directory / name
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid hardware research receipt")
        body = path.read_bytes()
        record = json.loads(body)
        if not isinstance(record, dict):
            raise ValueError("Hardware research receipt must be an object")
        records[name] = record
        references[name] = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}
    hardware, session = records["hardware.json"], records["slurm-session.json"]
    if expected_hardware is None and not expected_receipt_hashes:
        raise ValueError("Hardware research context needs an independent hardware/receipt binding")
    if expected_hardware is not None and hardware != expected_hardware:
        raise ValueError("Hardware research receipt differs from the authenticated runner")
    if expected_receipt_hashes is not None and any(expected_receipt_hashes.get(name) != ref["sha256"]
                                                  for name, ref in references.items()):
        raise ValueError("Hardware research receipt differs from its pinned archive")
    properties = hardware.get("device_properties")
    if (not isinstance(environment, dict) or not isinstance(properties, dict)
            or hardware.get("environment") != "romeo" or not session.get("session_id")
            or not str(expected_job_id).isdigit() or int(expected_job_id) <= 0
            or str(hardware.get("slurm_job_id")) != str(expected_job_id)
            or str(session.get("job_id")) != str(expected_job_id)
            or session["session_id"] != expected_session_id
            or session.get("uncertain") or "job_ids" in session
            or (expected_remote_dir is not None and session.get("remote_dir") != str(expected_remote_dir))):
        raise ValueError("Hardware research context lacks matching ROMEO ownership")
    physical_uuid = _uuid(properties.get("uuid"))
    if _uuid(environment.get("uuid")) != physical_uuid:
        raise ValueError("Hardware research context differs from the evaluator GPU")
    for key, value in environment.items():
        if key == "uuid":
            continue
        if properties.get(key) != value:
            raise ValueError("Hardware research context differs from evaluator " + key)
    rows = list(csv.reader(str(hardware.get("nvidia_smi", "")).splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("Hardware research context requires exactly one nvidia-smi GPU row")
    gpu, name, package, pci = [value.strip() for value in rows[0]]
    if (_uuid(gpu) != physical_uuid or not re.fullmatch(r"\d+(?:\.\d+){1,3}", package)
            or [_uuid(value) for value in hardware.get("gpu_uuids", [])] != [physical_uuid]):
        raise ValueError("Hardware research nvidia-smi identity is inconsistent")
    stable_keys = ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")
    if not all(key in hardware for key in stable_keys):
        raise ValueError("Hardware research fingerprint fields are incomplete")
    stable = {key: hardware[key] for key in stable_keys}
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    if fingerprint != hardware.get("fingerprint_id"):
        raise ValueError("Hardware research fingerprint differs from its archived bytes")
    return {"schema_version": 1, "kind": "authenticated_allocation_start_hardware_context",
        "scope": "Historical allocation-start observation; not a current GPU query or a compatibility decision",
        "hardware_currently_observed": False, "job_id": str(hardware["slurm_job_id"]),
        "session_id": session["session_id"], "gpu_uuid": physical_uuid,
        "machine": hardware["machine"], "hostname": hardware.get("hostname"),
        "nvidia_driver_package_version": package, "nvidia_smi_gpu_name": name, "pci_bus_id": pci,
        "native_reported_versions": {key: deepcopy(properties[key]) for key in
            ("driver_version", "nvrtc_version", "cuda_toolkit_version") if key in properties},
        "nvcc_version_output": hardware["toolkit"], "host_compiler": hardware["compiler"],
        "hardware_fingerprint_id": fingerprint, "evidence": references,
        "producer": "remote_worker.fingerprint: nvidia-smi --query-gpu=uuid,name,driver_version,pci.bus_id --format=csv,noheader",
        "interpretation": "The package release is separate from cuDriverGetVersion's encoded CUDA API version. Consult official platform/version documentation; no applicability condition is supplied or satisfied by this annex."}


def runner_hardware_research_context(runner, environment):
    """Return an annex only for a runner carrying the owned ROMEO receipts."""
    if getattr(runner, "environment", None) != "romeo":
        return None
    backend = getattr(runner, "backend", None)
    if backend is None or not isinstance(getattr(runner, "hardware", None), dict):
        raise ValueError("ROMEO research context needs the authenticated runner/backend ownership")
    return archived_hardware_research_context(runner.results_dir, environment,
        expected_job_id=backend.job_id, expected_session_id=backend.session_id,
        expected_remote_dir=backend.remote_dir, expected_hardware=runner.hardware)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _json_evidence(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or unsafe baseline runtime evidence")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Baseline runtime evidence changed during capture")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Baseline runtime evidence must be an object")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _resource_limits(properties):
    result = {}
    for key in ("max_threads_per_block", "max_shared_memory_per_block_bytes"):
        value = properties.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError("Missing measured target resource limit: " + key)
        result[key] = value
    for key in ("max_block_dimensions", "max_grid_dimensions"):
        value = properties.get(key)
        if (not isinstance(value, list) or len(value) != 3
                or any(type(item) is not int or item <= 0 for item in value)):
            raise ValueError("Missing measured target resource limit: " + key)
        result[key] = list(value)
    for key in ("max_shared_memory_per_block_optin_bytes", "max_shared_memory_per_multiprocessor_bytes",
                "warp_size", "multiprocessor_count"):
        if key in properties:
            if type(properties[key]) is not int or properties[key] <= 0:
                raise ValueError("Invalid measured target resource limit: " + key)
            result[key] = properties[key]
    return result


def baseline_runtime_research_context(worker_directory, environment, baseline_path, *,
        expected_contract_hash, baseline_candidate, expected_case_ids, benchmark_series,
        launches_per_series, expected_job_id, expected_session_id, expected_remote_dir=None,
        expected_hardware=None, expected_receipt_hashes=None, expected_baseline_sha256=None,
        request_receipt_directory=None):
    """Summarize already observed public baseline work, with no hardware query.

    This is documentary evidence for research, not an applicability decision or
    candidate qualification. Inputs, outputs, checkpoints and final suites are
    never opened. An archived caller must pin the baseline as well as ownership.
    """
    if expected_hardware is None and expected_baseline_sha256 is None:
        raise ValueError("Archived runtime context requires an independently pinned baseline")
    if not isinstance(environment, dict) or any(environment.get(key) is None for key in
            ("uuid", "architecture", "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")):
        raise ValueError("Baseline runtime context requires a complete evaluator toolchain identity")
    cases = list(expected_case_ids)
    if (not cases or len(cases) != len(set(cases))
            or any(not isinstance(case, str) or not case.startswith("search-") for case in cases)
            or type(benchmark_series) is not int or benchmark_series < 1
            or type(launches_per_series) is not int or launches_per_series < 1):
        raise ValueError("Runtime context requires the exact public search baseline protocol")
    baseline_path = Path(baseline_path).absolute()
    if baseline_path.name != "baseline-measurement.json" or "final-groups" in baseline_path.parts:
        raise ValueError("Only the public search baseline can supply runtime research context")
    allocation = archived_hardware_research_context(worker_directory, environment,
        expected_job_id=expected_job_id, expected_session_id=expected_session_id,
        expected_remote_dir=expected_remote_dir, expected_hardware=expected_hardware,
        expected_receipt_hashes=expected_receipt_hashes)
    hardware, hardware_ref = _json_evidence(Path(worker_directory) / "hardware.json")
    if hardware_ref != allocation["evidence"]["hardware.json"]:
        raise ValueError("Hardware receipt changed during baseline context capture")
    limits = _resource_limits(hardware["device_properties"])
    baseline_candidate.check_unchanged()
    source_hash = hashlib.sha256(baseline_candidate.source.read_bytes()).hexdigest()
    baseline, baseline_ref = _json_evidence(baseline_path)
    if expected_baseline_sha256 is not None and baseline_ref["sha256"] != expected_baseline_sha256:
        raise ValueError("Baseline runtime receipt differs from its independent pin")
    if (baseline.get("ok") is not True or baseline.get("status") != "measured"
            or baseline.get("candidate_id") != "baseline" or baseline_candidate.candidate_id != "baseline"
            or baseline.get("artifact_hash") != baseline_candidate.artifact_hash
            or baseline.get("contract_hash") != expected_contract_hash
            or not isinstance(baseline.get("evaluation_id"), str) or not baseline["evaluation_id"]
            or set(baseline.get("cases", {})) != set(cases)
            or baseline.get("protocol", {}).get("benchmark_series") != benchmark_series
            or baseline.get("protocol", {}).get("launches_per_series") != launches_per_series):
        raise ValueError("Baseline runtime artifact, contract or complete case protocol differs")

    def check_environment(actual):
        if not isinstance(actual, dict) or _uuid(actual.get("uuid")) != allocation["gpu_uuid"]:
            raise ValueError("Baseline runtime GPU differs from the owned allocation")
        if any(actual.get(key) != value for key, value in environment.items() if key != "uuid"):
            raise ValueError("Baseline runtime toolchain differs from the evaluator")

    def check_report(report):
        if (report.get("ok") is not True or report.get("stage") != "complete"
                or str(report.get("slurm_job_id")) != str(expected_job_id)
                or report.get("hardware_fingerprint") != hardware):
            raise ValueError("Baseline native report lacks the same successful owned allocation")
        check_environment(report.get("environment"))
        identity = report.get("compilation_identity", {})
        expected = {"architecture": environment["architecture"], "driver_version": environment["driver_version"],
                    "nvrtc_version": environment["nvrtc_version"], "toolkit_version": environment["cuda_toolkit_version"],
                    "harness_build_id": environment["harness_build_id"]}
        if (any(identity.get(key) != value for key, value in expected.items())
                or hashlib.sha256(identity.get("source", "").encode()).hexdigest() != source_hash
                or report.get("architecture") != environment["architecture"]):
            raise ValueError("Baseline native compilation identity differs from the exact source/toolchain")

    check_environment(baseline.get("environment"))
    reports, launches, compilation_observations = {}, {}, {}
    for case_id in cases:
        case = baseline["cases"][case_id]
        series = case.get("series", [])
        if len(series) != benchmark_series:
            raise ValueError("Baseline runtime series are incomplete")
        for series_index, row in enumerate(series):
            report = row.get("report", {})
            check_report(report)
            samples = report.get("latency_samples_ms", [])
            if (len(samples) != launches_per_series or row.get("raw_ms") != samples
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)
                    or row.get("mean_ms") != statistics.fmean(samples)
                    or report.get("guard_readonly_ok") is not True or report.get("guard_outputs_poisoned") is not True):
                raise ValueError("Baseline runtime lacks complete successful guarded native series")
            report_hash = _canonical_hash(report)
            if report_hash in reports:
                raise ValueError("A native series was repeated in the baseline receipt")
            reports[report_hash] = report
            launches[report_hash] = row.get("launch")
            if report.get("compile_cache_hit") is False:
                compilation_observations[report_hash] = {"receipt": baseline_ref,
                    "locator": f"cases/{case_id}/series/{series_index}/report",
                    "report_canonical_sha256": report_hash, "source_sha256": source_hash,
                    "compilation_identity_canonical_sha256": _canonical_hash(report["compilation_identity"]),
                    "compile_cache_hit": False, "stage": "complete"}
        if case.get("latency_ms") != statistics.median(row["mean_ms"] for row in series):
            raise ValueError("Baseline runtime aggregate differs from its native series")

    linked = {}
    if request_receipt_directory is not None:
        directory = Path(request_receipt_directory).resolve()
        if directory != Path(worker_directory).resolve().parent / "request-receipts":
            raise ValueError("Runtime request receipts must belong to the same campaign")
        for path in sorted(directory.rglob("*.json")):
            receipt, reference = _json_evidence(path)
            report = receipt.get("report", {})
            report_hash = _canonical_hash(report)
            source_matches = hashlib.sha256(report.get("compilation_identity", {}).get("source", "").encode()).hexdigest() == source_hash
            if report_hash not in reports and not (source_matches and report.get("ok") is True and report.get("compile_cache_hit") is False):
                continue
            check_report(report)
            request = receipt.get("request", {})
            if (request.get("kernel") != baseline_candidate.kernel
                    or request.get("compiler_options") != report["compilation_identity"].get("options")):
                raise ValueError("Baseline native receipt names a different kernel or compiler options")
            if report_hash in reports:
                launch = {key: request.get(key) for key in ("grid", "block", "shared_memory_bytes")}
                if launch != launches[report_hash]:
                    raise ValueError("Baseline native receipt launch differs from its measured series")
                linked[report_hash] = reference
            if report.get("compile_cache_hit") is False:
                compilation_observations[report_hash] = {"receipt": reference, "run_id": receipt.get("run_id"),
                    "report_canonical_sha256": report_hash, "source_sha256": source_hash,
                    "compilation_identity_canonical_sha256": _canonical_hash(report["compilation_identity"]),
                    "compile_cache_hit": False, "stage": "complete"}
        if set(linked) != set(reports):
            raise ValueError("Baseline series lack their exact public native request receipts")
    baseline_candidate.check_unchanged()
    if _json_evidence(baseline_path)[1] != baseline_ref:
        raise ValueError("Baseline runtime receipt changed during capture")
    if _json_evidence(Path(worker_directory) / "hardware.json")[1] != hardware_ref:
        raise ValueError("Hardware receipt changed during baseline context capture")
    if _json_evidence(Path(worker_directory) / "slurm-session.json")[1] != allocation["evidence"]["slurm-session.json"]:
        raise ValueError("Ownership receipt changed during baseline context capture")
    return {"schema_version": 1, "kind": "authenticated_baseline_runtime_observation",
        "scope": "Already observed ordinary baseline work on this exact allocation; no current hardware query or future compatibility decision",
        "hardware_currently_observed": False, "runtime_initialization_observed": True,
        "nvrtc_compilation_observed": bool(compilation_observations),
        "job_id": allocation["job_id"], "session_id": allocation["session_id"], "gpu_uuid": allocation["gpu_uuid"],
        "hardware_fingerprint_id": allocation["hardware_fingerprint_id"], "resource_limits": limits,
        "resource_limits_producer": "Native CUDA driver device attributes in the authenticated hardware.json",
        "native_reported_versions": allocation["native_reported_versions"],
        "baseline": {"evaluation_id": baseline["evaluation_id"], "artifact_hash": baseline["artifact_hash"],
            "source_sha256": source_hash, "contract_hash": expected_contract_hash, "case_ids": sorted(cases),
            "case_count": len(cases), "series_per_case": benchmark_series, "series_count": len(reports),
            "launches_per_series": launches_per_series, "all_native_series_ok": True,
            "all_native_input_output_guards_passed": True, "native_receipts_linked": len(linked),
            "native_report_hashes_canonical_sha256": _canonical_hash(sorted(reports)),
            "benchmark_validation_scope": "The measured benchmark producer runs independent validation before timing; this annex does not rerun an oracle or certify a new candidate"},
        "native_compilation_observations": [compilation_observations[key] for key in sorted(compilation_observations)],
        "evidence": {**allocation["evidence"], "baseline-measurement.json": baseline_ref,
                     "baseline_native_receipts": [linked[key] for key in sorted(linked)]},
        "limitations": ["VBIOS and GPU subrevision remain unknown unless separately measured; no value is inferred",
            "Observed ordinary compilation/execution does not qualify a new kernel, special architecture feature or future allocation",
            "No performance claim, profiling counters or applicability flag is assigned; normal research and evaluator guards remain authoritative",
            "No input/output buffer, private final suite, seed or checkpoint was opened"]}


def runner_runtime_research_context(runner, environment, baseline_path, *, expected_contract_hash,
        baseline_candidate, expected_case_ids, benchmark_series, launches_per_series):
    """Bind existing baseline facts to the same owned runner; perform no RPC."""
    if getattr(runner, "environment", None) != "romeo":
        return None
    backend = getattr(runner, "backend", None)
    hardware = getattr(runner, "hardware", None)
    if backend is None and not isinstance(hardware, dict):
        return None
    if backend is None or not isinstance(hardware, dict):
        raise ValueError("Runtime research context needs the authenticated runner/backend ownership")
    directory = Path(runner.results_dir).resolve().parent / "request-receipts"
    result = baseline_runtime_research_context(runner.results_dir, environment, baseline_path,
        expected_contract_hash=expected_contract_hash, baseline_candidate=baseline_candidate,
        expected_case_ids=expected_case_ids, benchmark_series=benchmark_series,
        launches_per_series=launches_per_series, expected_job_id=backend.job_id,
        expected_session_id=backend.session_id, expected_remote_dir=backend.remote_dir,
        expected_hardware=runner.hardware, request_receipt_directory=directory if directory.is_dir() else None)
    result["runner_binding"] = "Same authenticated owned runner hardware/job/session; no new device query"
    return result
