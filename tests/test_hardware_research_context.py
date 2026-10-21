"""Hardware package metadata supplements research without changing raw identity."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from optimizer_agent.hardware_research_context import (
    archived_hardware_research_context, runner_hardware_research_context,
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
