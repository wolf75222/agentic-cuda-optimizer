import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

import vanilla_gpu_service as module


class SimulatedRunner:
    """CPU-only harness double; these unit tests never submit a GPU job."""
    def __init__(self, directory):
        self.results_dir = Path(directory)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def get_device_properties(self, device):
        return {"name": "SIMULATED_TEST_DEVICE", "uuid": "cpu-unit-test-only", "architecture": "sm_90",
                "driver_version": 13010, "nvrtc_version": "12.6", "cuda_toolkit_version": 12060,
                "harness_build_id": "simulated", "max_threads_per_block": 1024,
                "max_shared_memory_per_block_bytes": 49152}

    def run(self, run_id, request):
        self.calls.append(request.mode)
        n = int(request.arguments[-1].value)
        a, b = (np.fromfile(request.arguments[index].input_file, dtype="<f4") for index in (0, 1))
        x = a.astype(np.float64)
        if request.kernel == "matmul":
            output = (x.reshape(n, n) @ b.astype(np.float64).reshape(n, n)).astype("<f4")
        elif request.kernel == "reduction":
            output = np.asarray([np.sum(x)], dtype="<f4")
        elif request.kernel == "scan":
            output = np.cumsum(x).astype("<f4")
        elif request.kernel == "stencil":
            padded = np.pad(x, (1, 1))
            output = (padded[:-2] + padded[1:-1] + padded[2:]).astype("<f4")
        elif request.kernel == "convolution":
            padded = np.pad(x, (1, 1))
            output = (padded[:-2] * b[0] + padded[1:-1] * b[1] + padded[2:] * b[2]).astype("<f4")
        elif request.kernel == "normalization":
            output = (x / np.sqrt(np.mean(x*x) + 1e-5)).astype("<f4")
        else:
            raise AssertionError("Unknown test operation")
        if "WRONG_OUTPUT" in Path(request.source).read_text():
            output[:] = 0
        path = self.results_dir / (run_id + ".bin")
        output.tofile(path)
        return {"ok": True, "environment": self.get_device_properties(0),
                "guard_readonly_ok": True, "guard_outputs_poisoned": True,
                "outputs": [{"path": str(path), "bytes": output.nbytes}],
                "latency_ms": 2.0, "latency_samples_ms": [2.0] * request.iterations}


@pytest.fixture
def service(tmp_path, monkeypatch, request):
    class Backend:
        def __init__(self, config, repo, directory, session_id):
            self.job_id = None
            self.simulated = SimulatedRunner(directory)
            self.closed = False
        def start(self):
            self.job_id = "12345"
        def wait_until_ready(self):
            pass
        def runner(self):
            return self.simulated
        def close(self, *, cancel=False):
            self.closed = True
        def wait_until_finished(self, timeout):
            return {"job_id": self.job_id, "terminal": True, "state": "COMPLETED", "exit_code": "0:0"}
    monkeypatch.setattr(module, "SlurmBackend", Backend)
    operation = getattr(request, "param", "matmul")
    config = module.CampaignConfig(run_id="literal-test", workload=operation, search_sizes=(128, 256, 512), benchmark_series=1,
                                   confirmation_batches=2, launches_per_series=2, warmup=0)
    value = module.VanillaGPUService(config, tmp_path / "protected", tmp_path / "child", "literal-test")
    value.start()
    (value.child_workspace / "kernel.cu").write_text(module.baseline_source(config))
    (value.child_workspace / "launch.json").write_text(json.dumps(module.workload_description(operation)["baseline_launch_rule"]))
    try:
        yield value
    finally:
        value.close()


def http(service, operation, body=None, *, token=None):
    request = urllib.request.Request(service.metadata["endpoint"] + "/" + operation,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + (token if token is not None else service.token),
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def test_service_exposes_fixed_search_contract_and_keeps_oracle_outside_child(service):
    result = http(service, "properties")
    assert result["ok"] and result["baseline"]["ok"]
    assert result["search_sizes"] == [128, 256, 512]
    assert result["validation_sizes"] == [1, 17, 63, 128, 256, 512]
    assert set(service.evaluator._suites) == {"short", "search", "performance"}
    assert not list(service.protected_dir.rglob("final-*"))
    assert not list(service.child_workspace.rglob("*.bin"))
    assert result["contract"]["allowed_precision"] == "float32"
    policy = json.loads(result["contract"]["metadata_json"])["memory_policy"]
    assert "disjoint" in policy["allocations"] and "double" in policy["arithmetic"]
    saved = json.loads((service.protected_dir / "service.json").read_text())
    assert "token" not in saved
    assert saved["token_sha256"]


def test_http_authentication_and_final_operations_cannot_change_evaluator(service):
    before = len(service.backend.simulated.calls)
    with pytest.raises(urllib.error.HTTPError) as failure:
        http(service, "properties", token="wrong")
    assert failure.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as failure:
        http(service, "final", {})
    assert failure.value.code == 400
    assert len(service.backend.simulated.calls) == before
    audit = (service.protected_dir / "requests.jsonl").read_text()
    assert "rejected" in audit and "invalid_request" in audit
    assert service.token not in audit


def test_exact_candidate_is_frozen_private_and_raw_search_series_are_recorded(service):
    result = http(service, "benchmark", {"source": "kernel.cu", "launch": "launch.json"})
    assert result["ok"] and result["result"]["status"] == "measured"
    assert set(result["result"]["cases"]) == {"search-128", "search-256", "search-512"}
    assert result["result"]["cases"]["search-128"]["series"][0]["raw_ms"] == [2.0, 2.0]
    snapshot = next((service.protected_dir / "candidates").glob("*/kernel.cu"))
    original = snapshot.read_text()
    (service.child_workspace / "kernel.cu").write_text("changed after request")
    assert snapshot.read_text() == original
    assert json.loads(Path(result["proof_path"]).read_text()) == result


def test_wrong_output_never_reaches_benchmark_and_double_is_rejected_before_harness(service):
    path = service.child_workspace / "kernel.cu"
    path.write_text(path.read_text() + "\n// WRONG_OUTPUT")
    before = len(service.backend.simulated.calls)
    wrong = http(service, "benchmark", {"source": "kernel.cu", "launch": "launch.json"})
    assert not wrong["ok"] and wrong["result"]["status"] == "numerical_error"
    assert set(service.backend.simulated.calls[before:]) == {"run"}
    path.write_text(path.read_text().replace("float sum", "double sum"))
    before = len(service.backend.simulated.calls)
    precision = http(service, "benchmark", {"source": "kernel.cu", "launch": "launch.json"})
    assert not precision["ok"] and precision["result"]["status"] == "contract_violation"
    assert len(service.backend.simulated.calls) == before


def test_candidate_paths_cannot_escape_child_or_change_configuration(service):
    before = len(service.backend.simulated.calls)
    for source in ("../protected/baseline.json", str(service.protected_dir / "baseline.json")):
        with pytest.raises(urllib.error.HTTPError) as failure:
            http(service, "validate", {"source": source, "launch": "launch.json"})
        assert failure.value.code == 400
    link = service.child_workspace / "link.cu"
    link.symlink_to(service.protected_dir / "baseline.json")
    with pytest.raises(urllib.error.HTTPError) as failure:
        http(service, "validate", {"source": "link.cu", "launch": "launch.json"})
    assert failure.value.code == 400
    with pytest.raises(urllib.error.HTTPError):
        http(service, "benchmark", {"source": "kernel.cu", "launch": "launch.json", "oracle": "changed"})
    assert len(service.backend.simulated.calls) == before


def test_http_measurements_are_serialized(service, monkeypatch):
    active = 0
    maximum = 0
    lock = threading.Lock()
    def validate(candidate, suite):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.05)
        with lock:
            active -= 1
        return {"ok": True, "status": "validated", "artifact_hash": candidate.artifact_hash}
    monkeypatch.setattr(service.evaluator, "validate", validate)
    results = []
    def evaluate():
        results.append(http(service, "validate", {"source": "kernel.cu", "launch": "launch.json"}))
    first, second = threading.Thread(target=evaluate), threading.Thread(target=evaluate)
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert len(results) == 2 and all(result["ok"] for result in results)
    assert maximum == 1


def test_tool_cli_uses_authenticated_loopback_and_preserves_response(service, capsys):
    path = module.REPO / "scripts/vanilla_gpu_tool.py"
    spec = importlib.util.spec_from_file_location("vanilla_gpu_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.main(["properties", "--endpoint", service.metadata["endpoint"], "--token", service.token]) == 0
    assert json.loads(capsys.readouterr().out)["properties"]["name"] == "SIMULATED_TEST_DEVICE"
    assert tool.main(["properties", "--endpoint", service.metadata["endpoint"], "--token", "wrong"]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_context_close_is_idempotent_and_job_scoped(service):
    receipt = service.close()
    assert receipt["job_id"] == "12345" and receipt["terminal"]
    assert service.backend.closed
    assert service.close() == receipt
    assert json.loads((service.protected_dir / "worker-terminal.json").read_text()) == receipt


def test_proof_directory_must_be_disjoint_from_child(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        module.VanillaGPUService(module.CampaignConfig(), tmp_path / "child/proof", tmp_path / "child", "new")


@pytest.mark.parametrize("service", module.OPERATIONS, indirect=True)
def test_all_six_operations_have_exact_math_abi_buffers_and_private_oracle(service):
    description = http(service, "properties")
    operation = service.config.workload
    assert description["operation"] == operation
    assert f"void {operation}(" in description["workload_description"]["signature"]
    assert description["contract"]["operation"] == operation
    assert description["workload_description"]["mathematics"]
    case = service.evaluator._suites["search"][-1]
    n = case.metadata["n"]
    assert case.request.arguments[0].bytes == 4 * (n*n if operation == "matmul" else n)
    assert case.request.arguments[1].bytes == 4 * (n*n if operation == "matmul" else 3 if operation == "convolution" else n)
    assert case.request.arguments[2].bytes == 4 * (1 if operation == "reduction" else n*n if operation == "matmul" else n)
    result = http(service, "validate", {"source": "kernel.cu", "launch": "launch.json"})
    assert result["ok"]
    assert {row["case_id"] for row in result["result"]["cases"]} == {
        f"search-{size}" for size in service.validation_sizes}
    assert not list(service.protected_dir.rglob("final-*"))


def test_borrowed_worker_is_never_submitted_or_stopped(tmp_path, monkeypatch):
    runner = SimulatedRunner(tmp_path / "caller-owned-worker")
    runner.backend = SimpleNamespace(job_id="caller-owned-job")
    monkeypatch.setattr(module, "SlurmBackend", lambda *args, **kwargs: pytest.fail("borrowed mode cannot create a backend"))
    config = module.CampaignConfig(run_id="borrowed-test", workload="scan", search_sizes=(64, 128),
                                  benchmark_series=1, launches_per_series=2, confirmation_batches=2, warmup=0)
    with module.VanillaGPUService(config, tmp_path / "proof", tmp_path / "child", "borrowed", runner=runner) as value:
        assert value.backend is None
        assert value.metadata["owns_worker"] is False
        assert value.metadata["slurm_job_id"] == "caller-owned-job"
        assert value.metadata["search_sizes"] == [64, 128]
        assert value.metadata["validation_sizes"] == [1, 17, 63, 64, 128]
    assert value.close() == {"job_id": "caller-owned-job", "ownership": "borrowed", "terminal": False,
                             "state": "BORROWED_WORKER_PRESERVED", "service_closed": True}


def test_native_gain_observer_is_private_fresh_once_and_keeps_midturn_tokens_unknown(tmp_path):
    runner = SimulatedRunner(tmp_path/"runner")
    config = module.CampaignConfig(run_id="native-observer",workload="scan",search_sizes=(8,),
        benchmark_series=1,launches_per_series=2,confirmation_batches=2,warmup=0)
    with module.VanillaGPUService(config,tmp_path/"proof",tmp_path/"child","native-observer",
            runner=runner,observe_first_gain=True,observer_started_at_unix=time.time()-5) as value:
        (value.child_workspace/"kernel.cu").write_text(module.baseline_source(config))
        (value.child_workspace/"launch.json").write_text(json.dumps(value.description["baseline_launch_rule"]))
        first = http(value,"benchmark",{"source":"kernel.cu","launch":"launch.json"})
        assert first["ok"] and "first_useful_gain" not in json.dumps(first)
        state = value.gain_observer.snapshot()
        receipt = next(iter(state["observations"].values()))
        assert receipt["decision"] == "exact_artifact_identity" and receipt["status"] == "identity_observed"
        assert receipt["confirmation"]["pairs"] == [] and "interval" not in receipt["confirmation"]
        requests = state["metrics"]["observer_gpu_requests"]
        assert requests == 0 and receipt["observer_cuda_event_kernel_seconds"] == 0
        second = http(value,"benchmark",{"source":"kernel.cu","launch":"launch.json"})
        assert second["result"]["evaluation_id"] != first["result"]["evaluation_id"]
        metrics = value.observer_metrics()
        assert metrics["observer_gpu_requests"] == requests
        assert metrics["observer_confirmations_completed"] == 0
        assert metrics["observer_identity_observations"] == 1
        assert metrics["method_consumption"]["method_gpu_evaluation_calls"] == 3
        assert metrics["method_consumption"]["model_tokens"] is None
        assert metrics["combined_method_and_observer_gpu_wall_seconds"] > metrics["observer_wall_seconds"]
        assert "first_useful_gain" not in json.dumps(http(value,"properties"))


def test_native_unproven_identity_stays_unknown_and_does_not_stop_method_benchmarks(tmp_path):
    class IncompleteIdentityRunner(SimulatedRunner):
        def get_device_properties(self,device):
            properties = super().get_device_properties(device)
            properties.pop("harness_build_id")
            return properties
    runner = IncompleteIdentityRunner(tmp_path/"runner")
    config = module.CampaignConfig(run_id="native-observer-unknown",workload="scan",search_sizes=(8,),
        benchmark_series=1,launches_per_series=2,confirmation_batches=2,warmup=0)
    with module.VanillaGPUService(config,tmp_path/"proof",tmp_path/"child","native-observer-unknown",
            runner=runner,observe_first_gain=True) as value:
        (value.child_workspace/"kernel.cu").write_text(module.baseline_source(config))
        (value.child_workspace/"launch.json").write_text(json.dumps(value.description["baseline_launch_rule"]))
        first = http(value,"benchmark",{"source":"kernel.cu","launch":"launch.json"})
        assert first["ok"] and "first_useful_gain" not in json.dumps(first)
        receipt = next(iter(value.gain_observer.snapshot()["observations"].values()))
        assert receipt["status"] == "identity_context_unproven"
        assert receipt["decision"] == "artifact_identity_context_unproven"
        second = http(value,"benchmark",{"source":"kernel.cu","launch":"launch.json"})
        assert second["ok"] and second["result"]["evaluation_id"] != first["result"]["evaluation_id"]
        metrics = value.observer_metrics()
        assert metrics["first_useful_gain_status"] == "unknown_identity_context"
        assert metrics["observer_gpu_requests"] == 0 and metrics["observer_confirmations_completed"] == 0
        assert metrics["observer_identity_observations"] == 0 and metrics["observer_identity_context_unproven"] == 1
        assert metrics["method_consumption"]["method_gpu_evaluation_calls"] == 3
        assert metrics["time_to_first_useful_gain_seconds"] is None


def test_observer_infrastructure_incident_preserves_measurement_then_stops_native_gpu(tmp_path,monkeypatch):
    runner = SimulatedRunner(tmp_path/"runner")
    config = module.CampaignConfig(run_id="native-observer-stop",workload="scan",search_sizes=(8,),
        benchmark_series=1,launches_per_series=2,confirmation_batches=2,warmup=0)
    with module.VanillaGPUService(config,tmp_path/"proof",tmp_path/"child","observer-stop",
            runner=runner,observe_first_gain=True) as value:
        (value.child_workspace/"kernel.cu").write_text(module.baseline_source(config))
        (value.child_workspace/"launch.json").write_text(json.dumps(value.description["baseline_launch_rule"]))
        def failure(*args,**kwargs):
            return {"ok":False,"status":"infrastructure_failure","quarantined":True}
        monkeypatch.setattr(value.gain_observer.evaluator,"confirm",failure)
        with pytest.raises(RuntimeError,match="infrastructure"):
            value.dispatch("benchmark",{"source":"kernel.cu","launch":"launch.json"},request_id="incident")
        assert json.loads((value.protected_dir/"requests/incident/response.json").read_text())["ok"]
        before = len(runner.calls)
        with pytest.raises(RuntimeError,match="stopped"):
            value.dispatch("validate",{"source":"kernel.cu","launch":"launch.json"})
        assert len(runner.calls) == before
        assert value.observer_metrics()["first_useful_gain_status"] == "unknown_incident"
