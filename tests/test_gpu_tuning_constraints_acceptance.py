"""CPU control tests with simulated CUDA reports; these prove no GPU execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/gpu_tuning_constraints_acceptance.py"
SPEC = importlib.util.spec_from_file_location("gpu_tuning_constraints_acceptance_script", MODULE)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

PROPERTIES = {"name": "CPU TEST SIMULATOR", "uuid": "simulated-not-a-gpu", "architecture": "sm_90",
    "driver_version": 13010, "nvrtc_version": "12.6", "cuda_toolkit_version": 12060,
    "harness_build_id": "simulated-build", "max_threads_per_block": 1024,
    "max_shared_memory_per_block_bytes": 49152, "max_block_dimensions": [1024, 1024, 64]}


def install_simulator(monkeypatch, tmp_path, *, lost_ack=False, wrong_job=False, wrong_resources=False):
    """Real evaluator/tuner, fake worker and timings explicitly confined to tests."""
    class Runner:
        results_dir = tmp_path / "simulated-worker"
        def __init__(self):
            self.calls = []
        def get_device_properties(self, device):
            assert device == 0
            return PROPERTIES
        def run(self, run_id, request):
            n = request.arguments[-1].value
            source = Path(request.source).read_text()
            matches = re.findall(r"(?m)^#define TILE ([0-9]+)$", source)
            tile = int(matches[0]) if matches else None
            assert tile != 12, "DSL-rejected TILE12 reached the worker"
            if tile is not None:
                assert request.block == [tile, tile, 1]
                assert request.grid == [(n + tile - 1) // tile] * 2 + [1]
            self.calls.append({"run_id": run_id, "mode": request.mode, "n": n, "tile": tile, "source": request.source})
            a, b = (np.fromfile(request.arguments[index].input_file, dtype="<f4").astype(np.float64).reshape(n, n)
                    for index in (0, 1))
            value = (a @ b).astype("<f4").ravel()
            output = tmp_path / (run_id + ".bin")
            value.tofile(output)
            latency = {None: 4.0, 8: 2.0, 16: 1.0}[tile]
            static_shared = 2 * tile * tile * 4 if tile else 0
            return {"ok": True, "environment": PROPERTIES, "slurm_job_id": "wrong-job" if wrong_job else "900303",
                "guard_readonly_ok": True, "guard_outputs_poisoned": True,
                "outputs": [{"path": str(output), "bytes": value.nbytes}],
                "latency_samples_ms": [latency] * request.iterations if request.mode == "benchmark" else [],
                "latency_ms": latency if request.mode == "benchmark" else None,
                "reservation_wall_seconds": 0.01, "compile_ms": 0.1, "compile_cache_hit": False,
                "kernel_resources": {"static_shared_memory_bytes": static_shared + int(wrong_resources)}}
    runner, instances = Runner(), []
    class Backend:
        hardware = {"device_properties": PROPERTIES}
        def __init__(self, config, repo, directory, *, session_id):
            self.config, self.session_id, self.directory = config, session_id, Path(directory)
            self.remote_dir = config.remote_root + "/sessions/" + session_id
            self.job_id, self.closed = None, False
            instances.append(self)
        def start(self):
            intent = json.loads((self.directory.parent / "allocation-intent.json").read_text())
            assert intent["session_id"] == self.session_id
            assert intent["remote_dir"] == self.remote_dir
            cpu_plan = json.loads((self.directory.parent / "preallocation-cpu-plan.json").read_text())
            assert [row["macro_value"] for row in cpu_plan["preflight"] if not row["dsl_admitted"]] == [12]
            if lost_ack:
                raise RuntimeError("Simulated submission ACK lost; exact session must be reconciled")
            self.job_id = "900303"
            return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return runner
        def close(self, *, cancel=False):
            assert self.job_id == "900303" and not cancel
            self.closed = True
        def wait_until_finished(self, timeout):
            return {"job_id": self.job_id, "state": "COMPLETED", "exit_code": "0:0", "terminal": True}
    monkeypatch.setattr(helper, "SlurmBackend", Backend)
    return runner, instances


def test_default_cpu_plan_materializes_distinct_macro_and_coupled_launch_without_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(helper, "SlurmBackend", lambda *a, **k: pytest.fail("CPU plan must not create SSH/Slurm"))
    assert helper.main(["--output", str(tmp_path), "--run-id", "cpu-plan"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["submitted"] == 0 and printed["model_calls"] == 0
    assert printed["admitted_tiles"] == [8, 16] and printed["filtered_tiles"] == [12]
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["hardware_verified"] is False
    assert plan["correctness_sizes"] == [1, 17, 63, 32, 64]
    assert {row["target"] for row in plan["parameter_bindings"]} == {"source_macro", "block_x", "block_y", "tile_x", "tile_y"}
    assert len({row["candidate"]["source_hash"] for row in plan["preflight"]}) == 3
    evaluator, _, template, *_ = helper.prepare(tmp_path, "cpu-plan")
    materializer = helper.Autotuner(evaluator, work_directory=tmp_path / "check")
    for row in plan["preflight"]:
        tile = row["macro_value"]
        assert row["candidate"]["parameters"] == {"TILE": tile}
        assert row["candidate"]["launch_rule"]["block"] == [tile, tile, 1]
        assert row["candidate"]["launch_rule"]["tile_x"] == row["candidate"]["launch_rule"]["tile_y"] == tile
        assert f"#define TILE {tile}\n" in row["candidate"]["source"]
        assert row["launches"]["63"]["grid"] == [(63 + tile - 1) // tile] * 2 + [1]
        candidate = materializer.materialize(template, {"TILE": tile})
        assert helper.hardware_preflight(candidate, PROPERTIES) is None
    assert not (tmp_path / "allocation-intent.json").exists()


def test_real_tuner_filters_before_worker_compiles_and_validates_admitted_sources_then_remeasures(tmp_path, monkeypatch):
    runner, backends = install_simulator(monkeypatch, tmp_path)
    external_config = tmp_path / "alternate-execution-config.json"
    external_config.write_bytes((helper.REPO / "configs/romeo.json").read_bytes())
    directory = tmp_path / "proof"
    result = helper.run_acceptance(directory, "simulated-dsl", external_config, benchmark_series=2, launches=2)
    assert result["ok"] and result["status"] == "dsl_source_macro_acceptance_passed"
    assert all(result["checks"].values())
    assert [(row["parameters"], row["status"]) for row in result["configurations"]] == [
        ({"TILE": 8}, "measured"), ({"TILE": 12}, "filtered"), ({"TILE": 16}, "measured")]
    assert result["configurations"][1]["measurement"] is None
    assert result["configurations"][1]["validation"] is None
    assert not [row for row in runner.calls if row["tile"] == 12]
    assert result["winner"]["parameters"] == {"TILE": 16}
    old_ids = {row["measurement"]["evaluation_id"] for row in result["configurations"] if row["measurement"]}
    assert result["winner_remeasurement"]["evaluation_id"] not in old_ids
    assert result["winner_remeasurement"]["artifact_hash"] == result["winner"]["artifact_hash"]
    assert result["ledger_usage"]["spent"]["local_configurations"] == 3
    assert result["ledger_usage"]["spent"]["compilations"] == 2
    assert result["local_search_controls"] == {"configurations": 3, "compilations": 2}
    for tile in (8, 16):
        calls = [row for row in runner.calls if row["tile"] == tile]
        first_benchmark = next(index for index, row in enumerate(calls) if row["mode"] == "benchmark")
        assert {row["n"] for row in calls[:first_benchmark]} == {1, 17, 63, 32, 64}
        assert len([row for row in calls if row["mode"] == "benchmark"]) == (8 if tile == 16 else 4)
    assert len({row["run_id"] for row in runner.calls}) == len(runner.calls)
    assert len(list((directory / "request-receipts").glob("*.json"))) == len(runner.calls)
    assert len(backends) == 1 and backends[0].closed
    assert backends[0].session_id.startswith("dsl-source-macro-")
    assert result["terminal"]["job_id"] == result["job_id"] == "900303"
    assert result["model_calls"] == result["model_tokens"] == 0 and result["excluded_from_quality_study"] is True
    hashes = result["source_provenance"]["sources"]
    assert hashes[str(external_config)] == hashlib.sha256(external_config.read_bytes()).hexdigest()
    support = helper.REPO / "scripts/gpu_autotuning_acceptance.py"
    assert hashes[str(support.relative_to(helper.REPO))] == hashlib.sha256(support.read_bytes()).hexdigest()
    for path in ("optimizer_agent/autotuning.py", "optimizer_agent/tuning_constraints.py", "optimizer_agent/campaign.py"):
        assert path in hashes
    with pytest.raises(ValueError, match="fresh experiment"):
        helper.run_acceptance(directory, "simulated-dsl", external_config)
    assert len(backends) == 1


def test_lost_submission_ack_preserves_exact_intent_and_prevents_duplicate_submission(tmp_path, monkeypatch):
    runner, backends = install_simulator(monkeypatch, tmp_path, lost_ack=True)
    directory = tmp_path / "lost-ack"
    receipt = helper.run_acceptance(directory, "lost-ack", helper.REPO / "configs/romeo.json", launches=2)
    assert not receipt["ok"] and "ACK lost" in receipt["error"]
    intent = json.loads((directory / "allocation-intent.json").read_text())
    assert intent["session_id"] == receipt["session_id"] == backends[0].session_id
    assert intent["remote_dir"] == receipt["remote_dir"]
    assert receipt["job_id"] is None and not runner.calls and not backends[0].closed
    with pytest.raises(ValueError, match="fresh experiment"):
        helper.run_acceptance(directory, "lost-ack", helper.REPO / "configs/romeo.json")
    assert len(backends) == 1


@pytest.mark.parametrize("defect,check", [
    ({"wrong_job": True}, "all_requests_owned_job"),
    ({"wrong_resources": True}, "native_compilation_or_content_cache_and_static_shared_evidence"),
])
def test_qualified_oracle_outputs_cannot_hide_wrong_job_or_wrong_compiled_macro_resources(tmp_path, monkeypatch, defect, check):
    install_simulator(monkeypatch, tmp_path, **defect)
    result = helper.run_acceptance(tmp_path / "bad-proof", "bad-proof", helper.REPO / "configs/romeo.json",
                                   benchmark_series=1, launches=1)
    assert result["checks"]["both_admitted_configurations_validated_and_measured"] is True
    assert result["checks"][check] is False
    assert result["ok"] is False and result["status"] == "incomplete"
