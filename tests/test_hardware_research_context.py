"""Hardware package metadata supplements research without changing raw identity."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from optimizer_agent.hardware_research_context import (
    archived_hardware_research_context, runner_hardware_research_context,
    baseline_runtime_research_context, runner_runtime_research_context,
)
from optimizer_agent.research_service import cuda_toolchain_documentation_requests


def receipts(tmp_path):
    properties = {"uuid": "a" * 32, "architecture": "sm_90", "driver_version": 13010,
        "nvrtc_version": "12.6", "cuda_toolkit_version": 12060, "harness_build_id": "test-build"}
    hardware = {"environment": "romeo", "hostname": "test-node", "machine": "aarch64",
        "slurm_job_id": "123", "device_properties": properties,
        "nvidia_smi": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa, Test GPU, 590.48.01, 0000:01:00.0",
        "gpu_uuids": ["GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        "toolkit": "Test nvcc release 12.6", "compiler": "Test compiler"}
    stable = {k: hardware[k] for k in ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")}
    hardware["fingerprint_id"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    session = {"job_id": "123", "session_id": "owned-test-session", "remote_dir": "/test/sessions/owned-test-session"}
    for name, value in (("hardware.json", hardware), ("slurm-session.json", session)):
        (tmp_path / name).write_text(json.dumps(value))
    return properties, hardware, session


def test_package_release_is_separate_authenticated_historical_context(tmp_path):
    properties, hardware, session = receipts(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    environment = deepcopy(properties)
    context = archived_hardware_research_context(tmp_path, environment, expected_job_id="123",
        expected_session_id="owned-test-session", expected_hardware=hardware)
    assert context["nvidia_driver_package_version"] == "590.48.01"
    assert context["native_reported_versions"]["driver_version"] == 13010
    assert context["machine"] == "aarch64" and context["hardware_currently_observed"] is False
    assert "no applicability condition" in context["interpretation"]
    assert environment == properties
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for name, ref in context["evidence"].items():
        assert ref["sha256"] == hashlib.sha256(before[name]).hexdigest()
    backend = SimpleNamespace(job_id="123", session_id="owned-test-session", remote_dir=session["remote_dir"])
    runner = SimpleNamespace(environment="romeo", results_dir=tmp_path, hardware=hardware, backend=backend)
    assert runner_hardware_research_context(runner, environment) == context
    assert runner_hardware_research_context(SimpleNamespace(environment="local"), environment) is None


@pytest.mark.parametrize("mutation", ["job", "uuid", "native_version", "package_row", "multiple_gpus", "fingerprint", "malformed", "symlink"])
def test_unbound_or_tampered_metadata_is_refused(tmp_path, mutation):
    properties, hardware, session = receipts(tmp_path)
    bound_hardware = deepcopy(hardware)
    expected = deepcopy(properties)
    if mutation == "job":
        session["job_id"] = "456"
    elif mutation == "uuid":
        expected["uuid"] = "b" * 32
    elif mutation == "native_version":
        expected["nvrtc_version"] = "13.0"
    elif mutation == "package_row":
        hardware["nvidia_smi"] = hardware["nvidia_smi"].replace("aaaaaaaa", "bbbbbbbb", 1)
    elif mutation == "multiple_gpus":
        hardware["nvidia_smi"] += "\n" + hardware["nvidia_smi"]
    elif mutation == "fingerprint":
        hardware["toolkit"] = "Changed compiler"
    for name, value in (("hardware.json", hardware), ("slurm-session.json", session)):
        (tmp_path / name).write_text(json.dumps(value))
    if mutation == "malformed":
        (tmp_path / "hardware.json").write_text("[]")
    if mutation == "symlink":
        target = tmp_path / "hardware.json"
        target.rename(tmp_path / "other.json")
        target.symlink_to(tmp_path / "other.json")
    with pytest.raises(ValueError):
        archived_hardware_research_context(tmp_path, expected, expected_job_id="123",
            expected_session_id="owned-test-session", expected_hardware=bound_hardware)


@pytest.mark.parametrize("field,value", [("job_id", "456"), ("job_id", "0"), ("session_id", "foreign"),
    ("uncertain", True), ("job_ids", ["123"]), ("remote_dir", "/foreign")])
def test_runner_ownership_cannot_be_replaced_by_another_consistent_pair(tmp_path, field, value):
    properties, hardware, session = receipts(tmp_path)
    backend = SimpleNamespace(job_id="123", session_id="owned-test-session", remote_dir=session["remote_dir"])
    session[field] = value
    if field == "job_id":
        hardware["slurm_job_id"] = value  # job is intentionally outside stable hardware fingerprint
    (tmp_path / "slurm-session.json").write_text(json.dumps(session))
    (tmp_path / "hardware.json").write_text(json.dumps(hardware))
    runner = SimpleNamespace(environment="romeo", results_dir=tmp_path, hardware=hardware, backend=backend)
    with pytest.raises(ValueError):
        runner_hardware_research_context(runner, properties)


def test_toolchain_sources_are_versioned_real_reads_not_applicability_assertions():
    workload = {"environment": {"driver_version": 13010, "cuda_toolkit_version": 12060}}
    before = deepcopy(workload)
    sources = cuda_toolchain_documentation_requests(workload)
    assert len(sources) == 3
    assert "archive/12.6.0/cuda-driver-api/group__CUDA__VERSION.html" in sources[0]["url"]
    assert "archive/12.6.0/cuda-toolkit-release-notes/" in sources[1]["url"]
    assert "minor-version-compatibility.html" in sources[2]["url"]
    assert all(item["kind"] == "cuda_documentation" and "satisfied" not in item for item in sources)
    assert workload == before
    assert cuda_toolchain_documentation_requests({"environment": {"driver_version": 13010}}) == []


def runtime_fixture(tmp_path):
    """CPU-only fabricated receipts exercise identity checks, never CUDA."""
    from optimizer_agent.evaluation import Candidate, LaunchRule

    worker = tmp_path / "worker"
    worker.mkdir()
    properties, hardware, session = receipts(worker)
    properties.update(max_threads_per_block=1024, max_block_dimensions=[1024, 1024, 64],
        max_grid_dimensions=[2147483647, 65535, 65535], max_shared_memory_per_block_bytes=49152,
        max_shared_memory_per_block_optin_bytes=232448, max_shared_memory_per_multiprocessor_bytes=233472)
    stable = {k: hardware[k] for k in ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")}
    hardware["fingerprint_id"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    (worker / "hardware.json").write_text(json.dumps(hardware))
    environment = {key: value for key, value in properties.items() if not key.startswith("max_")}
    generation = tmp_path / "generation"
    generation.mkdir()
    source = generation / "baseline.cu"
    source.write_text('extern "C" __global__ void test_kernel(float* a, int n) { if (threadIdx.x < n) a[threadIdx.x] = 0; }')
    candidate = Candidate("baseline", source, "test_kernel", LaunchRule(block=(128, 1, 1)))
    directory = tmp_path / "request-receipts"
    directory.mkdir()
    launch = {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory_bytes": 0}
    cases = {}
    for case_index, case_id in enumerate(("search-1", "search-17")):
        series = []
        for series_index in range(2):
            report = {"ok": True, "stage": "complete", "id": f"native-{case_index}-{series_index}",
                "slurm_job_id": "123", "hardware_fingerprint": deepcopy(hardware),
                "environment": deepcopy(environment), "architecture": "sm_90", "compile_cache_hit": True,
                "compilation_identity": {"source": source.read_text(), "architecture": "sm_90", "driver_version": 13010,
                    "nvrtc_version": "12.6", "toolkit_version": 12060, "harness_build_id": "test-build", "options": []},
                "guard_readonly_ok": True, "guard_outputs_poisoned": True,
                "latency_samples_ms": [1.0, 2.0], "started_at_unix_ms": 1000 + case_index * 10 + series_index,
                "finished_at_unix_ms": 1001 + case_index * 10 + series_index}
            series.append({"report": report, "launch": deepcopy(launch), "raw_ms": [1.0, 2.0], "mean_ms": 1.5})
            request = {"kernel": candidate.kernel, "compiler_options": [], **deepcopy(launch),
                       "arguments": [{"input_file": str(tmp_path / "private-final-input.bin")} ]}
            (directory / (report["id"] + ".json")).write_text(json.dumps({"run_id": report["id"], "report": report, "request": request}))
        cases[case_id] = {"series": series, "latency_ms": 1.5}
    baseline = {"ok": True, "status": "measured", "candidate_id": "baseline", "artifact_hash": candidate.artifact_hash,
        "contract_hash": "contract-test", "evaluation_id": "benchmark-test", "environment": deepcopy(environment),
        "protocol": {"benchmark_series": 2, "launches_per_series": 2}, "cases": cases}
    path = generation / "baseline-measurement.json"
    path.write_text(json.dumps(baseline))
    compile_report = deepcopy(next(iter(cases.values()))["series"][0]["report"])
    compile_report.update(id="compile-observation", compile_cache_hit=False)
    (directory / "compile-observation.json").write_text(json.dumps({"run_id": "compile-observation",
        "report": compile_report, "request": {"kernel": candidate.kernel, "compiler_options": []}}))
    (tmp_path / "private-final-input.bin").write_bytes(b"must never be read by the annex")
    kwargs = {"expected_contract_hash": "contract-test", "baseline_candidate": candidate,
        "expected_case_ids": ["search-1", "search-17"], "benchmark_series": 2, "launches_per_series": 2,
        "expected_job_id": "123", "expected_session_id": session["session_id"],
        "expected_remote_dir": session["remote_dir"], "expected_hardware": deepcopy(hardware),
        "request_receipt_directory": directory}
    return worker, environment, path, baseline, kwargs


def test_runtime_annex_reports_only_observed_facts_and_never_queries_or_reads_buffers(tmp_path, monkeypatch):
    from pathlib import Path

    worker, environment, path, baseline, kwargs = runtime_fixture(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    read_bytes = Path.read_bytes

    def guarded_read(p):
        if p.suffix == ".bin" or p.suffix == ".sqlite":
            raise AssertionError("The documentary annex cannot open inputs or databases")
        return read_bytes(p)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    context = baseline_runtime_research_context(worker, environment, path, **kwargs)
    assert context["resource_limits"]["max_threads_per_block"] == 1024
    assert context["resource_limits"]["max_shared_memory_per_block_bytes"] == 49152
    assert context["runtime_initialization_observed"] is True and context["nvrtc_compilation_observed"] is True
    assert context["hardware_currently_observed"] is False
    assert context["baseline"]["series_count"] == 4 and context["baseline"]["native_receipts_linked"] == 4
    assert len(context["native_compilation_observations"]) == 1
    assert context["native_compilation_observations"][0]["run_id"] == "compile-observation"
    serialized = json.dumps(context)
    assert "latency" not in serialized and "latency_samples_ms" not in serialized
    assert "extern" not in serialized and "must never be read" not in serialized and '"satisfied"' not in serialized
    assert "VBIOS" in serialized and "future allocation" in serialized
    for ref in context["evidence"]["baseline_native_receipts"]:
        assert ref["sha256"] == hashlib.sha256(read_bytes(Path(ref["path"]))).hexdigest()
    assert before == {str(p.relative_to(tmp_path)): read_bytes(p) for p in tmp_path.rglob("*") if p.is_file()}

    def no_rpc(*args, **kwargs):
        raise AssertionError("No RPC is allowed")

    backend = SimpleNamespace(job_id="123", session_id=kwargs["expected_session_id"], remote_dir=kwargs["expected_remote_dir"])
    runner = SimpleNamespace(environment="romeo", results_dir=worker, hardware=kwargs["expected_hardware"], backend=backend,
                             run=no_rpc, get_device_properties=no_rpc)
    public = {key: kwargs[key] for key in ("expected_contract_hash", "baseline_candidate", "expected_case_ids",
                                         "benchmark_series", "launches_per_series")}
    wrapped = runner_runtime_research_context(runner, environment, path, **public)
    assert wrapped["baseline"] == context["baseline"] and "no new device query" in wrapped["runner_binding"]
    assert runner_runtime_research_context(SimpleNamespace(environment="local"), environment, path, **public) is None


@pytest.mark.parametrize("mutation", ["contract", "artifact", "source", "case", "series", "samples", "uuid", "job",
    "toolchain", "source_in_native", "native_launch", "native_kernel", "native_options", "missing_native_receipt", "missing_resource"])
def test_runtime_annex_refuses_unbound_partial_or_changed_evidence(tmp_path, mutation):
    worker, environment, path, baseline, kwargs = runtime_fixture(tmp_path)
    row = baseline["cases"]["search-1"]["series"][0]
    report = row["report"]
    native_path = kwargs["request_receipt_directory"] / "native-0-0.json"
    if mutation == "contract":
        baseline["contract_hash"] = "foreign"
    elif mutation == "artifact":
        baseline["artifact_hash"] = "foreign"
    elif mutation == "source":
        kwargs["baseline_candidate"].source.write_text("changed source")
    elif mutation == "case":
        del baseline["cases"]["search-17"]
    elif mutation == "series":
        baseline["cases"]["search-1"]["series"].pop()
    elif mutation == "samples":
        report["latency_samples_ms"].pop()
    elif mutation == "uuid":
        report["environment"]["uuid"] = "b" * 32
    elif mutation == "job":
        report["slurm_job_id"] = "456"
    elif mutation == "toolchain":
        report["compilation_identity"]["nvrtc_version"] = "13.0"
    elif mutation == "source_in_native":
        report["compilation_identity"]["source"] = "foreign source"
    elif mutation in {"native_launch", "native_kernel", "native_options"}:
        receipt = json.loads(native_path.read_text())
        if mutation == "native_launch":
            receipt["request"]["block"] = [64, 1, 1]
        elif mutation == "native_kernel":
            receipt["request"]["kernel"] = "foreign_kernel"
        else:
            receipt["request"]["compiler_options"] = ["--use_fast_math"]
        native_path.write_text(json.dumps(receipt))
    elif mutation == "missing_native_receipt":
        native_path.unlink()
    elif mutation == "missing_resource":
        hardware = kwargs["expected_hardware"]
        del hardware["device_properties"]["max_threads_per_block"]
        stable = {k: hardware[k] for k in ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")}
        hardware["fingerprint_id"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
        (worker / "hardware.json").write_text(json.dumps(hardware))
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        baseline_runtime_research_context(worker, environment, path, **kwargs)


def test_runtime_annex_does_not_invent_nvrtc_compile_from_cache_hits(tmp_path):
    worker, environment, path, baseline, kwargs = runtime_fixture(tmp_path)
    (kwargs["request_receipt_directory"] / "compile-observation.json").unlink()
    context = baseline_runtime_research_context(worker, environment, path, **kwargs)
    assert context["runtime_initialization_observed"] is True
    assert context["nvrtc_compilation_observed"] is False and context["native_compilation_observations"] == []


def test_archived_runtime_requires_independent_baseline_pin_and_rejects_final_cases(tmp_path):
    worker, environment, path, baseline, kwargs = runtime_fixture(tmp_path)
    hardware = kwargs.pop("expected_hardware")
    kwargs["expected_receipt_hashes"] = {name: hashlib.sha256((worker / name).read_bytes()).hexdigest()
                                        for name in ("hardware.json", "slurm-session.json")}
    with pytest.raises(ValueError, match="pinned baseline"):
        baseline_runtime_research_context(worker, environment, path, **kwargs)
    kwargs["expected_baseline_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    context = baseline_runtime_research_context(worker, environment, path, **kwargs)
    assert context["baseline"]["evaluation_id"] == "benchmark-test"
    kwargs["expected_baseline_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="independent pin"):
        baseline_runtime_research_context(worker, environment, path, **kwargs)
    kwargs["expected_case_ids"] = ["final-1"]
    with pytest.raises(ValueError, match="public search baseline"):
        baseline_runtime_research_context(worker, environment, path, **kwargs)


def test_runtime_wrapper_omits_absent_fixture_evidence_but_refuses_partial_ownership(tmp_path):
    arguments = {"expected_contract_hash": "not-read", "baseline_candidate": None,
                 "expected_case_ids": [], "benchmark_series": 1, "launches_per_series": 1}
    absent = SimpleNamespace(environment="romeo")
    assert runner_runtime_research_context(absent, {}, tmp_path / "missing.json", **arguments) is None
    for partial in (SimpleNamespace(environment="romeo", backend=object()),
                    SimpleNamespace(environment="romeo", hardware={})):
        with pytest.raises(ValueError, match="ownership"):
            runner_runtime_research_context(partial, {}, tmp_path / "missing.json", **arguments)
