"""Bootstrap-prefix recovery policy with simulated runners; no GPU or SSH."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from optimizer_agent.bootstrap_replay import BootstrapReplayError, BootstrapReplayRunner, create_manifest
from optimizer_agent.suite_orchestrator import RecordingRunner
from optimizer_agent.use_harness import BufferArgument, BufferDirection, KernelRequest


HARDWARE = {"slurm_job_id": "12345", "hostname": "simulated-test-node",
            "device_properties": {"uuid": "simulated-gpu", "architecture": "sm_90",
                                  "driver_version": 13010, "harness_build_id": "test-build"}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def report(directory, run_id, value=1.0):
    output = directory / (run_id + ".bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"done")
    return {"ok": True, "hardware_fingerprint": deepcopy(HARDWARE), "slurm_job_id": "12345",
            "environment": deepcopy(HARDWARE["device_properties"]),
            "outputs": [{"path": str(output), "bytes": 4}],
            "latency_ms": value, "latency_samples_ms": [value, value + .125],
            "latency_series_ms": [[value, value + .125]],
            "measurement_protocol": {"iterations": 2, "series": 1, "warmup": 3}}


class FakeRunner:
    def __init__(self, directory, *, lost_ack=False):
        self.directory = directory
        self.hardware = deepcopy(HARDWARE)
        self.calls, self.executions, self.responses = [], [], {}
        self.lost_ack = lost_ack
        self.before = None

    def run(self, run_id, request):
        self.calls.append((run_id, asdict(request)))
        if self.before:
            self.before(run_id, request)
        if run_id not in self.responses:
            self.executions.append(run_id)
            self.responses[run_id] = report(self.directory, run_id, value=7.0)
            if self.lost_ack:
                self.lost_ack = False
                raise TimeoutError("Simulated lost acknowledgement after execution")
        return deepcopy(self.responses[run_id])

    def get_device_properties(self, device=0):
        return self.hardware["device_properties"]


def prefix(tmp_path, *, interrupted=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "kernel.cu"
    source.write_text('extern "C" __global__ void f(const float* a,float* b) {}')
    inp = tmp_path / "input.bin"
    inp.write_bytes(b"data")
    dependency = tmp_path / "common.h"
    dependency.write_text("#define STRIDE 1\n")
    request = KernelRequest(str(source), "f", "run", 0, [1, 1, 1], [32, 1, 1], 0, 0, 1,
                            [BufferArgument(4, str(inp), BufferDirection.Input),
                             BufferArgument(4, None, BufferDirection.Output)],
                            dependencies=[str(dependency)], cache_dir=str(tmp_path / "compile-cache"))
    requests = [request, replace(request, mode="benchmark", warmup=3, iterations=2),
                replace(request, mode="benchmark", warmup=3, iterations=7)]
    receipts = tmp_path / "request-receipts"
    for index, current in enumerate(requests):
        value = {"run_id": f"original-{index}", "request": asdict(current),
                 "started_at_unix": 10.0 * index, "ended_at_unix": 10.0 * index + 1}
        if interrupted and index == 2:
            value["error"] = "Simulated SSH acknowledgement unavailable"
        else:
            value["report"] = report(tmp_path / "old-outputs", f"original-{index}", value=index + 1)
        # Filenames are deliberately in reverse lexical order.
        write_json(receipts / f"{2-index}.json", value)
    return receipts, requests, create_manifest(receipts, deepcopy(HARDWARE))


def test_prefix_order_raw_reports_exact_reconciliation_and_then_fresh_calls(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    for index, request in enumerate(requests):
        result = replay.run(f"new-random-{index}", request)
        if index < 2:
            assert result == manifest["entries"][index]["report"]
            assert runner.calls == []
    assert runner.calls == [("original-2", manifest["entries"][2]["original_request"])]
    assert replay.replay_position == 3 and replay.replay_complete
    assert replay.get_device_properties() == HARDWARE["device_properties"]
    # Identical semantics after the prefix are fresh measurements, every time.
    replay.run("fresh-a", requests[-1])
    replay.run("fresh-b", requests[-1])
    assert [row[0] for row in runner.calls] == ["original-2", "fresh-a", "fresh-b"]
    assert runner.executions == ["original-2", "fresh-a", "fresh-b"]


def test_semantic_paths_can_move_but_failed_envelope_uses_original_paths(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    source = tmp_path / "relocated.cu"
    source.write_bytes(Path(requests[0].source).read_bytes())
    inp = tmp_path / "relocated.bin"
    inp.write_bytes(Path(requests[0].arguments[0].input_file).read_bytes())
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    for index, request in enumerate(requests):
        moved = deepcopy(request)
        moved.source = str(source)
        moved.arguments[0].input_file = str(inp)
        moved.cache_dir = str(tmp_path / "different-host-cache")
        replay.run(f"new-{index}", moved)
    assert runner.calls == [("original-2", manifest["entries"][2]["original_request"])]


def test_lost_ack_is_journaled_before_call_and_recovery_reuses_original_id_once(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs", lost_ack=True)
    audit = tmp_path / "audit"
    def prepared(run_id, request):
        state = json.loads((audit / "position-000002.json").read_text())
        assert state["status"] == "prepared" and state["original_run_id"] == run_id == "original-2"
        assert asdict(request) == manifest["entries"][2]["original_request"]
    runner.before = prepared
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    replay.run("discarded-a", requests[0]); replay.run("discarded-b", requests[1])
    with pytest.raises(TimeoutError, match="lost acknowledgement"):
        replay.run("discarded-c", requests[2])
    assert replay.replay_position == 2
    assert json.loads((audit / "position-000002.json").read_text())["status"] == "awaiting_reconciliation"
    resumed = BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    for index, request in enumerate(requests):
        resumed.run(f"yet-another-{index}", request)
    assert [item[0] for item in runner.calls] == ["original-2", "original-2"]
    assert runner.executions == ["original-2"]
    assert json.loads((audit / "position-000002.json").read_text())["attempts"] == 2
    # A second full bootstrap replay reuses the durable recovered response.
    third = BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    for index, request in enumerate(requests):
        third.run(f"third-{index}", request)
    assert len(runner.calls) == 2


def test_composes_with_recording_runner_using_distinct_recovery_receipts(tmp_path):
    receipts, requests, manifest = prefix(tmp_path)
    original_bytes = {p.name: p.read_bytes() for p in receipts.glob("*.json")}
    remote = FakeRunner(tmp_path / "new-outputs")
    recording = RecordingRunner(remote, tmp_path / "recovery-receipts")
    replay = BootstrapReplayRunner(recording, manifest, HARDWARE, tmp_path / "audit")
    for index, request in enumerate(requests):
        replay.run(f"new-{index}", request)
    recorded = json.loads((tmp_path / "recovery-receipts/original-2.json").read_text())
    assert recorded["run_id"] == "original-2" and recorded["request"] == manifest["entries"][2]["original_request"]
    assert {p.name: p.read_bytes() for p in receipts.glob("*.json")} == original_bytes
    assert replay.hardware == HARDWARE


@pytest.mark.parametrize("changed", ["source", "input", "dependency", "output", "receipt"])
def test_changed_frozen_artifact_fails_before_any_runner_call(tmp_path, changed):
    receipts, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    target = {"source": Path(requests[0].source), "input": Path(requests[0].arguments[0].input_file),
              "dependency": Path(requests[0].dependencies[0]),
              "output": Path(manifest["entries"][0]["report"]["outputs"][0]["path"]),
              "receipt": receipts / "2.json"}[changed]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(BootstrapReplayError):
        replay.run("unused", requests[0])
    assert runner.calls == [] and replay.replay_position == 0


@pytest.mark.parametrize("changed", ["block", "warmup", "iterations", "series", "compiler_options", "guard_outputs", "sentinel_byte"])
def test_entire_request_protocol_is_bound_before_replay(tmp_path, changed):
    _, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    bad = deepcopy(requests[0])
    setattr(bad, changed, {"block": [64, 1, 1], "warmup": 7, "iterations": 2, "series": 2,
                           "compiler_options": ["--use_fast_math"], "guard_outputs": False, "sentinel_byte": 0}[changed])
    with pytest.raises(BootstrapReplayError, match="order/content mismatch"):
        replay.run("unused", bad)
    assert runner.calls == [] and replay.replay_position == 0


def test_request_order_mismatch_does_not_search_a_generic_cache(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    with pytest.raises(BootstrapReplayError, match="position 0"):
        replay.run("unused", requests[1])
    assert runner.calls == []


@pytest.mark.parametrize("changed", ["gpu", "job", "runner_later", "report_environment"])
def test_hardware_binding_refuses_other_allocation_or_native_environment(tmp_path, changed):
    receipts, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    expected = deepcopy(HARDWARE)
    if changed == "gpu":
        expected["device_properties"]["uuid"] = "other-gpu"
    elif changed == "job":
        expected["slurm_job_id"] = "54321"
    elif changed == "report_environment":
        path = receipts / "2.json"; value = json.loads(path.read_text())
        value["report"]["environment"]["uuid"] = "other-real-device"
        write_json(path, value)
        with pytest.raises(BootstrapReplayError, match="Native report hardware"):
            create_manifest(receipts, HARDWARE)
        return
    if changed == "runner_later":
        replay = BootstrapReplayRunner(runner, manifest, expected, tmp_path / "audit")
        runner.hardware["device_properties"]["uuid"] = "other-gpu"
        with pytest.raises(BootstrapReplayError, match="allocation/GPU"):
            replay.run("unused", requests[0])
    else:
        with pytest.raises(BootstrapReplayError, match="allocation/GPU"):
            BootstrapReplayRunner(runner, manifest, expected, tmp_path / "audit")
    assert runner.calls == []


@pytest.mark.parametrize("issue", ["two_failures", "failure_not_last", "duplicate_time", "overlap", "duplicate_id", "report_and_error"])
def test_ambiguous_or_nonserial_prefix_is_rejected(tmp_path, issue):
    receipts, _, _ = prefix(tmp_path)
    path = receipts / "2.json"; value = json.loads(path.read_text())
    if issue in ("two_failures", "failure_not_last"):
        value.pop("report"); value["error"] = "earlier interrupted request"
        if issue == "failure_not_last":
            last = receipts / "0.json"; row = json.loads(last.read_text()); row.pop("error")
            row["report"] = report(tmp_path / "old-outputs", "original-2")
            write_json(last, row)
    elif issue == "duplicate_time":
        value["started_at_unix"] = 10.0; value["ended_at_unix"] = 11.0
    elif issue == "overlap":
        value["ended_at_unix"] = 10.5
    elif issue == "duplicate_id":
        value["run_id"] = "original-1"
    else:
        value["error"] = "ambiguous"
    write_json(path, value)
    with pytest.raises(BootstrapReplayError):
        create_manifest(receipts, HARDWARE)


def test_manifest_sha_file_journal_and_output_snapshots_are_enforced(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest_path, HARDWARE, tmp_path / "audit")
    value = replay.run("new", requests[0])
    value["latency_ms"] = -999  # A caller cannot mutate stored proof by aliasing.
    journal = tmp_path / "audit/position-000000.json"
    state = json.loads(journal.read_text()); state["report"]["latency_ms"] = -999
    write_json(journal, state)
    second = BootstrapReplayRunner(runner, manifest_path, HARDWARE, tmp_path / "audit")
    with pytest.raises(BootstrapReplayError, match="journal SHA256 changed"):
        second.run("other", requests[0])
    changed = deepcopy(manifest); changed["entries"][0]["report"]["latency_ms"] = -999
    with pytest.raises(BootstrapReplayError, match="manifest SHA256"):
        BootstrapReplayRunner(runner, changed, HARDWARE, tmp_path / "new-audit")
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(BootstrapReplayError, match="bootstrap manifest artifact"):
        replay.run("other", requests[1])
    assert runner.calls == []


def test_prefix_without_transport_failure_never_calls_runner_until_exhausted(tmp_path):
    _, requests, manifest = prefix(tmp_path, interrupted=False)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit")
    with pytest.raises(BootstrapReplayError, match="before the frozen request prefix"):
        replay.require_complete()
    for request in requests:
        replay.run("ignored-new-uuid", request)
    replay.require_complete()
    assert runner.calls == [] and replay.replay_complete


def test_reconciled_output_mutation_blocks_resume_without_another_request(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    runner = FakeRunner(tmp_path / "new-outputs")
    audit = tmp_path / "audit"
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    for request in requests:
        last = replay.run("ignored-new", request)
    Path(last["outputs"][0]["path"]).write_bytes(b"evil")
    resumed = BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    resumed.run("ignored-new", requests[0]); resumed.run("ignored-new", requests[1])
    with pytest.raises(BootstrapReplayError, match="journal/artifacts changed"):
        resumed.run("ignored-new", requests[2])
    assert len(runner.calls) == 1 and resumed.replay_position == 2


@pytest.mark.parametrize("issue", ["missing_uuid", "zero_job", "missing_native_environment", "output_symlink"])
def test_incomplete_identity_or_unsafe_artifact_cannot_admit_a_prefix(tmp_path, issue):
    receipts, _, _ = prefix(tmp_path)
    hardware = deepcopy(HARDWARE)
    if issue == "missing_uuid":
        hardware["device_properties"].pop("uuid")
    elif issue == "zero_job":
        hardware["slurm_job_id"] = "0"
    else:
        path = receipts / "2.json"; value = json.loads(path.read_text())
        if issue == "missing_native_environment":
            value["report"].pop("environment")
            write_json(path, value)
        else:
            output = Path(value["report"]["outputs"][0]["path"])
            real = output.with_suffix(".real")
            output.rename(real); output.symlink_to(real)
    with pytest.raises(BootstrapReplayError):
        create_manifest(receipts, hardware)


def test_replay_journal_cannot_be_rebound_to_a_different_prefix(tmp_path):
    _, _, manifest = prefix(tmp_path / "first")
    _, _, other = prefix(tmp_path / "second")
    runner = FakeRunner(tmp_path / "new-outputs")
    audit = tmp_path / "audit"
    BootstrapReplayRunner(runner, manifest, HARDWARE, audit)
    with pytest.raises(BootstrapReplayError, match="journal manifest changed|another frozen manifest"):
        BootstrapReplayRunner(runner, other, HARDWARE, audit)
    assert runner.calls == []


def remove_request_files(manifest):
    saved = {}
    for entry in manifest["entries"]:
        for item in entry["request_files"]:
            path = Path(item["path"])
            if path not in saved:
                saved[path] = path.read_bytes()
                path.unlink()
    return saved


def test_explicit_defer_accepts_missing_request_files_until_exact_regeneration(tmp_path):
    _, requests, manifest = prefix(tmp_path)
    saved = remove_request_files(manifest)
    runner = FakeRunner(tmp_path / "new-outputs")
    # Missing files remain fatal under the default policy.
    with pytest.raises(BootstrapReplayError, match="Missing or unsafe"):
        BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "default-audit")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "deferred-audit",
                                   defer_request_files=True)
    assert runner.calls == []
    for path, content in saved.items():
        path.write_bytes(content)
    for request in requests:
        replay.run("new-ignored-id", request)
    replay.require_complete()
    assert runner.calls == [("original-2", manifest["entries"][2]["original_request"])]


@pytest.mark.parametrize("role", ["source", "input", "dependency"])
@pytest.mark.parametrize("regenerated", [False, True])
def test_deferred_missing_or_changed_request_files_fail_on_first_run_before_gpu(tmp_path, role, regenerated):
    _, requests, manifest = prefix(tmp_path)
    saved = remove_request_files(manifest)
    runner = FakeRunner(tmp_path / "new-outputs")
    replay = BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit", defer_request_files=True)
    for path, content in saved.items():
        path.write_bytes(content)
    target = Path(next(item["path"] for item in manifest["entries"][0]["request_files"] if item["role"] == role))
    if regenerated:
        target.write_bytes(target.read_bytes() + b"different regenerated bytes")
    else:
        target.unlink()
    with pytest.raises(BootstrapReplayError):
        replay.run("unused-new-id", requests[0])
    assert runner.calls == [] and replay.replay_position == 0
    assert not list((tmp_path / "audit").glob("position-*.json"))


@pytest.mark.parametrize("changed", ["receipt", "output", "hardware"])
def test_defer_never_skips_receipt_output_or_hardware_checks_at_construction(tmp_path, changed):
    receipts, _, manifest = prefix(tmp_path)
    remove_request_files(manifest)
    runner = FakeRunner(tmp_path / "new-outputs")
    if changed == "receipt":
        path = receipts / "2.json"
        path.write_text(path.read_text() + " ")
    elif changed == "output":
        Path(manifest["entries"][0]["report"]["outputs"][0]["path"]).write_bytes(b"evil")
    else:
        runner.hardware["device_properties"]["uuid"] = "different-gpu"
    with pytest.raises(BootstrapReplayError):
        BootstrapReplayRunner(runner, manifest, HARDWARE, tmp_path / "audit", defer_request_files=True)
    assert runner.calls == []
