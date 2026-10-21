"""Acceptance-helper control tests; all worker observations here are simulated."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/gpu_autotuning_acceptance.py"
SPEC = importlib.util.spec_from_file_location("gpu_autotuning_acceptance_script", MODULE)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_default_plan_submits_no_worker_and_materializes_exact_launch_bindings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "SlurmBackend", lambda *a, **k: pytest.fail("CPU plan cannot create an SSH/GPU backend"))
    assert helper.main(["--output", str(tmp_path), "--run-id", "plan-only", "--launches", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["submitted"] == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["correctness_sizes"] == [1, 17, 63, 64, 128, 256]
    assert plan["parameter_space"] == {"BLOCK_X": [8, 16, 32]}
    assert plan["model_calls"] == 0
    evaluator, baseline, template, domain, _ = helper.prepare(tmp_path, "plan-only")
    tuner = helper.Autotuner(evaluator, work_directory=tmp_path / "cpu-materialized")
    candidates = [tuner.materialize(template, {"BLOCK_X": x}) for x in domain["BLOCK_X"]]
    assert [c.launch_rule.block for c in candidates] == [(8, 8, 1), (16, 8, 1), (32, 8, 1)]
    assert all(c._source_hash == baseline._source_hash for c in candidates)
    assert [c.launch_rule.apply({"n": 63})["grid"] for c in candidates] == [[8, 8, 1], [4, 8, 1], [2, 8, 1]]


def test_helper_uses_real_tuner_api_fresh_winner_and_owned_terminal_receipt(tmp_path, monkeypatch):
    properties = {"name": "UNIT TEST SIMULATOR", "uuid": "simulated-unit", "architecture": "sm_90",
                  "driver_version": 13010, "nvrtc_version": "12.6", "cuda_toolkit_version": 12060,
                  "harness_build_id": "simulated-build", "max_threads_per_block": 1024,
                  "max_shared_memory_per_block_bytes": 49152}
    class Runner:
        results_dir = tmp_path / "simulated-worker"
        def __init__(self):
            self.calls, self.cached = [], {}
        def get_device_properties(self, device):
            return properties
        def run(self, run_id, request):
            n = request.arguments[-1].value
            self.calls.append((request.source, request.mode, n))
            key = request.arguments[0].input_file
            if key not in self.cached:
                a, b = (np.fromfile(request.arguments[i].input_file, dtype="<f4").astype(np.float64).reshape(n, n) for i in (0, 1))
                self.cached[key] = (a @ b).astype("<f4").ravel()
            output = tmp_path / (run_id + ".bin")
            self.cached[key].tofile(output)
            latency = {8: 2.0, 16: 1.0, 32: 1.5}[request.block[0]]
            return {"ok": True, "environment": properties, "guard_readonly_ok": True, "guard_outputs_poisoned": True,
                    "outputs": [{"path": str(output), "bytes": self.cached[key].nbytes}],
                    "latency_samples_ms": [latency] * request.iterations if request.mode == "benchmark" else [],
                    "latency_ms": latency if request.mode == "benchmark" else None,
                    "reservation_wall_seconds": 0.01, "compile_ms": 0.1, "compile_cache_hit": False}
    runner, instances = Runner(), []
    class Backend:
        hardware = {"device_properties": properties}
        def __init__(self, *args, **kwargs):
            self.job_id, self.closed, self.session_id = None, False, kwargs["session_id"]
            instances.append(self)
        def start(self):
            self.job_id = "900101"
            return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return runner
        def close(self, *, cancel=False):
            assert self.job_id == "900101"
            self.closed = True
        def wait_until_finished(self, timeout):
            return {"job_id": self.job_id, "state": "COMPLETED", "exit_code": "0:0", "terminal": True}
    monkeypatch.setattr(helper, "SlurmBackend", Backend)
    result = helper.run_acceptance(tmp_path / "acceptance", "test-autotune", helper.REPO / "configs/romeo.json",
                                   benchmark_series=2, launches=2)
    assert result["ok"] and result["status"] == "autotuning_acceptance_passed"
    assert all(result["checks"].values())
    assert result["winner"]["parameters"] == {"BLOCK_X": 16}
    assert result["ledger_usage"]["spent"]["local_configurations"] == 3
    assert result["ledger_usage"]["spent"]["compilations"] == 3
    assert result["local_search_controls"] == {"configurations": 3, "compilations": 3}
    assert "wall_seconds" not in result["local_search_controls"]
    assert result["model_calls"] == 0
    ids = {row["measurement"]["evaluation_id"] for row in result["configurations"]}
    assert result["winner_remeasurement"]["evaluation_id"] not in ids
    for candidate in result["candidates"].values():
        calls = [row for row in runner.calls if row[0] == candidate["source_path"]]
        first_benchmark = next(i for i, row in enumerate(calls) if row[1] == "benchmark")
        assert {row[2] for row in calls[:first_benchmark]} == {1, 17, 63, 64, 128, 256}
    assert len(instances) == 1 and instances[0].closed
    assert instances[0].session_id.startswith("ac12-autotune-")
    assert (tmp_path / "acceptance/autotuning-acceptance.json").is_file()
    assert len(list((tmp_path / "acceptance/request-receipts").glob("*.json"))) == len(runner.calls)
    with pytest.raises(ValueError, match="fresh experiment"):
        helper.run_acceptance(tmp_path / "acceptance", "test-autotune", helper.REPO / "configs/romeo.json")
    assert len(instances) == 1  # Existing proof and worker cannot be reused accidentally.
